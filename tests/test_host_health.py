from otrader.host_health import should_pause, should_reduce


def test_memory_warning_reduces_and_critical_pauses():
 assert should_reduce({'memory_pressure':'warning'})
 assert not should_pause({'memory_pressure':'warning'})
 assert should_pause({'memory_pressure':'critical'})
 assert not should_reduce({'memory_pressure':'normal'})


def test_reported_thermal_throttle_pauses_without_assuming_missing_sensor():
 assert should_pause({'cpu_speed_limit_pct':60})
 assert should_pause({'thermal_warning':True})
 assert not should_pause({'cpu_speed_limit_pct':None,'thermal_warning':False})
