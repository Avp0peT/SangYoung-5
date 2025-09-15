CHANGES & UPLOAD REPORT

已执行操作：

1. 新增文件：`.gitignore` — 忽略模型、数据集、临时缓存等不应上传的内容。
2. 删除：`checkpoints/`, `test_results/`, `trained_test_results/`, `uploads/`, `__pycache__/`（如果存在）。
3. 在本地提交了变更（commit message: "chore: add .gitignore and remove large artifacts from repo"）。

推送到远程结果：

- 使用 HTTPS 推送时失败：
  错误消息示例：
  "Failed to connect to 127.0.0.1 port 7890 after 2110 ms: Could not connect to server"
  这通常表明当前环境的网络/代理设置阻止了访问 GitHub（常见于公司网络或需要代理时）。

- 尝试切换为 SSH 并推送时失败：
  错误消息示例：
  "git@github.com: Permission denied (publickey)."
  表明本机未配置或未加载 SSH 私钥到当前会话，或该 SSH key 未绑定到 GitHub 账户。

建议的解决步骤：

1) 如果你使用代理：
   - 确认代理正在运行并且地址端口正确；或在临时会话中移除代理变量后重试。
   - 在 PowerShell 中（示例）：
     ```powershell
     $env:HTTP_PROXY = 'http://127.0.0.1:7890'
     $env:HTTPS_PROXY = 'http://127.0.0.1:7890'
     git push origin main
     ```

2) 使用 SSH（推荐）：
   - 在本机生成 SSH key（若无）：
     ```powershell
     ssh-keygen -t ed25519 -C "your_email@example.com"
     # 然后将生成的公钥 (~/.ssh/id_ed25519.pub) 的内容复制到 GitHub -> Settings -> SSH and GPG keys
     ```
   - 验证 SSH 连接：
     ```powershell
     ssh -T git@github.com
     ```
   - 确认后推送：
     ```powershell
     git push origin main
     ```

3) 如果你无法修复网络/SSH：
   - 你可以把仓库目录用外部工具手动上传（不推荐），或把压缩包交给能访问网络的机器进行推送。

记录与元信息：
- 本地提交已完成；需要你在具有网络访问或已正确配置 SSH 的环境中运行 `git push origin main`。


生成此报告的时间：2025-09-15
