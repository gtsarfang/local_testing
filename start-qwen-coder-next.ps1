# Adjust these two paths for your machine:
$LlamaServer = "C:\Users\George\.unsloth\llama.cpp\build\bin\Release\llama-server.exe"
$Model = "C:\Users\George\models\Qwen3-Coder-Next-UD-Q4_K_XL.gguf"

# Slower than start-qwen-coder.ps1 (see README: Model comparison) - use for
# problems where the extra quality is worth the wait, not as the default.
& $LlamaServer `
  -m $Model `
  --host 127.0.0.1 --port 8080 `
  -ngl 999 -ncmoe 41 `
  -c 32768 -fa on `
  -ctk q8_0 -ctv q8_0 `
  --no-mmap `
  --jinja `
  -np 1
