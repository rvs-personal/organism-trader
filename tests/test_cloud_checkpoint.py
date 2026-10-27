import json
import sqlite3
import zipfile
import pytest
from otrader.cloud_checkpoint import checkpoint, restore
from otrader.agents.snapshot import StateDB


def test_checkpoint_includes_committed_wal_and_excludes_uncommitted_changes(tmp_path):
    run = tmp_path / 'original'
    db = StateDB(run / 'state.sqlite')
    db.set_state('clock', '18')
    (run / 'pause_after_day.json').write_text('{"day":50}')
    (run / '.env').write_text('PRIVATE_SECRET=never-export')
    db.conn.execute('BEGIN')
    db.conn.execute("UPDATE run_state SET value='19' WHERE key='clock'")
    archive = checkpoint(run, tmp_path / 'persistent/latest.zip')
    restore(archive, tmp_path / 'restored')
    with sqlite3.connect(tmp_path / 'restored/state.sqlite') as copy:
        assert copy.execute("SELECT value FROM run_state WHERE key='clock'").fetchone()[0] == '18'
    assert not (tmp_path / 'restored/.env').exists()
    assert json.loads((tmp_path / 'restored/pause_after_day.json').read_text()) == {'day': 50}
    db.conn.rollback()
    db.conn.close()
    with pytest.raises(ValueError, match='overwrite'):
        restore(archive, tmp_path / 'restored')


def test_corrupt_or_traversing_checkpoint_cannot_be_restored(tmp_path):
    run = tmp_path / 'original'
    db = StateDB(run / 'state.sqlite')
    db.set_state('clock', '1')
    db.conn.close()
    archive = checkpoint(run, tmp_path / 'valid.zip')
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(tmp_path/'bad.zip', 'w') as out:
        for name in src.namelist():
            out.writestr(name, b'corrupt' if name == 'state.sqlite' else src.read(name))
    with pytest.raises(ValueError, match='checksum'):
        restore(tmp_path/'bad.zip', tmp_path/'bad-run')
    assert not (tmp_path/'bad-run').exists()
    with zipfile.ZipFile(tmp_path/'traversal.zip', 'w') as out:
        out.writestr('../escape', 'x')
    with pytest.raises(ValueError, match='Unexpected'):
        restore(tmp_path/'traversal.zip', tmp_path/'escape-run')
    assert not (tmp_path/'escape').exists()


def test_checkpoint_preserves_review_and_intervention_but_not_private_files(tmp_path):
    run = tmp_path/'run'
    db = StateDB(run/'state.sqlite'); db.set_state('intervention:1', '{"approved":true}'); db.conn.close()
    (run/'reviews').mkdir()
    (run/'intervention-1.json').write_text('{"approved":true}')
    (run/'reviews/day-050-review.md').write_text('Reviewed')
    (run/'reviews/private.env').write_text('secret')
    (run/'reviews/secret.json').symlink_to(run/'reviews/private.env')
    checkpoint(run,tmp_path/'saved.zip'); restore(tmp_path/'saved.zip',tmp_path/'restored')
    assert (tmp_path/'restored/intervention-1.json').read_text() == '{"approved":true}'
    assert (tmp_path/'restored/reviews/day-050-review.md').read_text() == 'Reviewed'
    assert not (tmp_path/'restored/reviews/private.env').exists()
    assert not (tmp_path/'restored/reviews/secret.json').exists()
