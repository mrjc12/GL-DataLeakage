# Anonymous GitHub 配置说明（DataLeakage）

私有仓库推送成功后，按以下步骤生成**双盲审稿用**匿名链接。

## 1. 登录

1. 打开 https://anonymous.4open.science/
2. 点击 **Sign in with GitHub**（用你的 GitHub 账号 mrjc12 登录即可）
3. 授权后进入 Dashboard

> 服务只会**读取**仓库，不会修改你的 GitHub 代码。

## 2. 新建匿名镜像

1. 点击 **Anonymize** / **New repository**
2. **Repository URL** 填写（私有仓库）：
   ```
   https://github.com/mrjc12/DataLeakage
   ```
3. **Terms to redact**（每行一个，按需增删）：
   ```
   mrjc12
   secadm
   @.*\.(edu|com|cn)
   ```
   并补充：真实姓名、学校/单位英文名、邮箱、论文标题、项目主页 URL 等。
4. **Expiration**：按会议要求（审稿结束后 Redirect 或 Remove）
5. 若有会议 **Conference ID**，在表单中填入
6. 提交后得到链接，形如：
   ```
   https://anonymous.4open.science/r/xxxxxxxx
   ```
   将此链接写入论文附录，**不要**写真实 GitHub 仓库地址。

## 3. 自检

- 用无痕窗口打开匿名链接，确认 README、注释、`pipeline.svg` 无身份信息
- 大于 8MB 的文件在 Anonymous GitHub 上可能无法在线预览（`.pt`、部分 `.jsonl`），审稿人可下载 ZIP 或在 README 中按说明本地生成

## 4. 注意事项

- 匿名链接指向的是**镜像**，不是你的私有仓库 URL
- 投稿期间勿将同一仓库改为 **public**，否则可能去匿名
- 后续 `git push` 更新私有仓库后，匿名镜像会自动同步（无需重新创建）
