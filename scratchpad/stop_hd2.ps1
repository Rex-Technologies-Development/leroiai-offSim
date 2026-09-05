# Clean-stop the hd2 training: kill worker/launcher shells FIRST (so the loop can't respawn), then the
# training python, retried to beat the respawn race. Use this instead of TaskStop (which orphans the child).
for ($i=0; $i -lt 4; $i++) {
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(bash|sh)\.exe$' -and $_.CommandLine -match 'hd2_train|parallel_worker' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*tenure.train*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }
  Start-Sleep -Milliseconds 800
}
Start-Sleep -Seconds 2
$py = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*tenure.train*' })
$sh = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(bash|sh)\.exe$' -and $_.CommandLine -match 'hd2_train|parallel_worker' })
Write-Output "STOPPED. training python=$($py.Count)  worker shells=$($sh.Count)"
& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader