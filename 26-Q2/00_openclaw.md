# OpenClaw Startup

具体的部署过程可以参考[这里](https://github.com/6eanut/weekend-things/blob/main/202603-2122/things.md)

## skills

```
npx clawhub@latest install sonoscli --registry=https://cn.clawhub-mirror.com

https://clawhub.ai/pskoett/self-improving-agent
https://clawhub.ai/oswalpalash/ontology
https://clawhub.ai/matrixy/agent-browser-clawdbot
https://clawhub.ai/guohongbin-git/skill-finder-cn
https://clawhub.ai/ivangdavila/productivity
https://clawhub.ai/othmanadi/planning-with-files
https://clawhub.ai/wscats/code-analysis-skills
https://clawhub.ai/ivangdavila/skill-finder
https://clawhub.ai/russellfei/file-manager
https://clawhub.ai/rubenfb23/arxiv-watcher
```

## webchat api

会话生命周期管理(用于防止资源浪费)

* /session idle time(20m)
* /session max-age time(4h)

子智能体管理

* /subagents list
* /subagents kill id

## framework

/root/.openclaw/agents/main/sessions/sessions.json 记录了当前的session会话有哪些，体现在webchat里面，可以删除掉不用的
