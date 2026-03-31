$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

if (!(Test-Path ".\.venv\Scripts\python.exe")) {
  & python -m venv .venv
  $py = ".\.venv\Scripts\python.exe"
}

& $py -m pip install --upgrade pip
& $py -m pip install -r requirements\dev.txt
& $py -m pip install -r requirements\api.txt
