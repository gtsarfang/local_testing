# Adjust these two paths for your machine:
$LlamaServer = "C:\Users\George\.unsloth\llama.cpp\build\bin\Release\llama-server.exe"
$Model = "C:\Users\George\models\Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"

& $LlamaServer `
  -m $Model `
  --host 127.0.0.1 --port 8080 `
  -ngl 999 -ncmoe 27 `
  -c 32768 -fa on `
  -ctk q8_0 -ctv q8_0 `
  --no-mmap `
  --jinja `
  -np 1
