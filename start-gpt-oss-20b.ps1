# Adjust these two paths for your machine:
$LlamaServer = "C:\Users\George\.unsloth\llama.cpp\build\bin\Release\llama-server.exe"
$Model = "C:\Users\George\models\gpt-oss-20b-UD-Q4_K_XL.gguf"

# Fastest raw tok/s of the three (see README: Model comparison), but a
# reasoning model - it can spend a lot of tokens thinking before answering,
# so wall-clock time-to-answer isn't always the fastest in practice.
& $LlamaServer `
  -m $Model `
  --host 127.0.0.1 --port 8080 `
  -ngl 999 -ncmoe 4 `
  -c 32768 -fa on `
  -ctk q8_0 -ctv q8_0 `
  --no-mmap `
  --jinja `
  -np 1
