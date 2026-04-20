# Claude Code Startup

## debug

```
如果遇到：
litellm.exceptions.APIConnectionError: litellm.APIConnectionError: ModelstudioException - AsyncCompletions.create() got an unexpected keyword argument 'output_config'. Received Model Group=Qwen3.5-397B-A17B
修改experimental_pass_through/adapters/handler.py里面的：
completion_response = await litellm.acompletion(**completion_kwargs)
成为
completion_kwargs.pop("output_config", None)
completion_response = await litellm.acompletion(**completion_kwargs)

如果遇到：
litellm.proxy.route_llm_request.ProxyModelNotFoundError: 400: {'error': 'anthropic_messages: Invalid model name passed in model=claude-haiku-4-5-20251001. Call `/v1/models` to view available models for your key.'}
向config.yaml里面添加：
- model_name: claude-haiku-4-5-20251001
  litellm_params:
    model: modelstudio/DeepSeek-V3.2
    api_key: key
    api_base: "url"
```

## 使用技巧

```
# 在Claude正在干活的时候插一个问题进去，但这个问题不会被加入对话历史。
/btw 你在干啥？

# 弹出菜单让你选，是只回退代码还是只回退对话
/rewind

# 生成一份HTML报告，分析你过去一个月使用Claude Code的习惯，包括你最常用哪些命令，你有哪些重复性的操作模式，然后给你推荐一些自定义命令和Skills。
/insights

# Claude Code会同时启动三个平行的Agent，分别从代码复用、代码质量、运行效率三个角度审查你的改动
# 每次跟Claude code对话了很多轮，写了几个大的功能更新之后，都顺手跑一遍/simplify
/simplify

# Claude刚帮你梳理完一个方案的思路，你想沿着这个思路试两种不同的实现方式，/branch一下，两个会话各走一边，最后挑效果好的那个
/branch

# 让Claude定时重复执行某个任务
# 比如/loop 5m 检查一下部署状态，它会每五分钟帮你跑一次，不用你自己盯着，默认间隔是10分钟
/loop

# 在终端里打/rc，或者打完整的/remote-control，它会生成一个URL
# 用手机打开这个链接，你的整个Claude Code会话就出现在手机上了
# 电脑和手机是完全同步的，你在手机上发一条指令，终端那边也能看到，你在终端上操作，手机上也会实时更新
# 代码始终在你的电脑上跑，手机只是一个遥控器，你的文件系统、MCP服务器、项目配置，全部还在本地，手机只是给你提供了一个远程操作的窗口
/rc

# 当前的整段对话就会被导出成一个Markdown文件
/export

# 输入框内想要换行怎么办
先输入 \，再按 Enter
```

## 环境配置

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install 'litellm[proxy]'

curl -fsSL https://claude.ai/install.sh | bash

# write config.yaml
litellm_settings:
   drop_params: true

model_list:
- model_name: DeepSeek-V3.2
  litellm_params:
    model: modelstudio/DeepSeek-V3.2
    api_key: key
    api_base: "url"
- model_name: GLM-5-Turbo
  litellm_params:
    model: modelstudio/GLM-5-Turbo
    api_key: key
    api_base: "url"

# write  ~/.local/lib/python3.10/site-packages/litellm/llms/openai_like/providers.json
  "modelstudio": {
    "base_url": "url",
    "api_key_env": "key",
    "supported_endpoints": ["/v1/chat/completions"]
  }

export LITELLM_MASTER_KEY="sk-1234567890"

litellm --config config.yaml

curl -X POST http://0.0.0.0:4000/v1/chat/completions \
-H "Authorization: Bearer $LITELLM_MASTER_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "DeepSeek-V3.2",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'

export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"

claude --model DeepSeek-V3.2
```

## 附：

claude code的安装需要科学上网，服务器没法科学上网咋办？只要本地笔记本可以就行

```
# on desktop
ssh -R 17890:127.0.0.1:7890 youruser@your-server

# on server
export http_proxy=http://127.0.0.1:17890
export https_proxy=http://127.0.0.1:17890

curl -I https://claude.ai/install.sh
```
