$signature = @"
using System;
using System.Runtime.InteropServices;
public static class SleepUtil {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

Add-Type -TypeDefinition $signature -ErrorAction SilentlyContinue

$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1
$endAt = (Get-Date).AddHours(12)

while ((Get-Date) -lt $endAt) {
    [SleepUtil]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
    Start-Sleep -Seconds 30
}

[SleepUtil]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
