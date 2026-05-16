# 项目备份与恢复指南

本项目已经纳入 Git 版本管理，并推送到远程仓库。日常修改代码后，建议及时提交并推送，避免本地代码混乱后无法恢复。

## 查看当前版本

查看当前分支和文件状态：

```powershell
git status
```

查看最近的提交记录：

```powershell
git log --oneline --decorate -5
```

查看当前远程仓库：

```powershell
git remote -v
```

## 提交新版本

每次完成一轮可运行的修改后，在项目根目录执行：

```powershell
git status
git add .
git commit -m "描述这次修改"
git push
```

提交信息建议写清楚改了什么，例如：

```powershell
git commit -m "Fix video detection record display"
```

## 恢复到上一次提交

如果只是想撤销某个文件的本地修改：

```powershell
git restore 路径/文件名
```

如果想撤销所有未提交的本地修改，让代码回到上一次提交状态：

```powershell
git restore .
```

如果有已经暂存但还没有提交的内容，先取消暂存：

```powershell
git restore --staged .
git restore .
```

## 代码乱了时的恢复方式

查看有哪些文件被修改：

```powershell
git status
```

恢复单个文件：

```powershell
git restore 路径/文件名
```

恢复整个项目到最近一次提交：

```powershell
git restore .
```

如果已经提交了错误版本，但还没有推送，想回到上一个提交：

```powershell
git reset --hard HEAD~1
```

注意：`git reset --hard` 会丢弃本地未保存的修改，执行前应先确认 `git status`。

## 从远程仓库重新克隆项目

在新的目录中执行：

```powershell
git clone <远程仓库地址>
cd <项目目录>
```

本项目当前远程仓库可以通过下面命令查看：

```powershell
git remote -v
```

## 大文件说明

上传视频、识别输出视频、运行日志、虚拟环境、缓存、训练数据集和训练输出目录已经通过 `.gitignore` 排除，不建议直接提交到 Git。

模型权重、数据集压缩包和视频文件如果超过几十 MB，建议使用以下方式保存：

- Git LFS：适合需要和代码一起版本管理的模型文件。
- GitHub/Gitee Release：适合发布固定版本模型或演示文件。
- 网盘、Kaggle、Hugging Face、对象存储：适合大型数据集和大量实验输出。

