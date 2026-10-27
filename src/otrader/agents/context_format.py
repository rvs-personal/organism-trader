"""Lossless presentation changes; no retrieval, truncation or shared agent state."""
import json


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def research_json(research, compact=False):
    if not compact:
        return json.dumps(research)
    result = dict(research)
    rows = research.get('candidates', [])
    if rows:
        columns = list(rows[0])
        if all(set(row) == set(columns) for row in rows):
            result['candidates'] = {
                'format': 'Each row lists values in columns order; all candidates are retained.',
                'columns': columns,
                'rows': [[row[column] for column in columns] for row in rows],
            }
    return dumps(result)


def observer(db, agent_id, day, phase, enabled):
    if not enabled:
        return None
    def record(event):
        db.conn.execute('INSERT INTO inference_events(agent_id,sim_day,phase,event_json) VALUES (?,?,?,?)',
                        (agent_id, day, phase, dumps(event)))
        db._commit()
        print(dumps(dict(agent=agent_id, day=day, phase=phase, **event)), flush=True)
    return record
