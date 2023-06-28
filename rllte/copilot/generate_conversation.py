import g4f
import os
import re
import json

from g4f.Provider import (
    Ails,
    You,
    Bing,
    Yqcloud,
    Theb,
    Aichat,
    Bard,
    Vercel,
    Forefront,
    Lockchat,
    Liaobots,
    H2o,
    ChatgptLogin,
    DeepAi,
    GetGpt
)

# content = """
# Below is a conversation sample that introduces an assistant:
# [Human]: Who are you?
# [GPT]: I am Richard, an intelligent assistant powered by the long-term evolution project of reinforcement learning (RLLTE).
# [Human]: What can you do?
# [GPT]: I can help you build RL applications with rllte and answer RL questions.

# Please generate some similar conversation samples that have the same meaning.
# """

content = """
Below is a conversation sample that runs an RL application:
[Human]: I want to train an PPO agent on the Atari games.
[GPT]: \nOkay, you can use the following command:\n``` sh\npython -m rllte.hub.apps.ppo_atari --env-id PongNoFrameskip-v4\n```\n

Please generate some similar conversation samples that have the same meaning.
"""

# get a response from the model
response = g4f.ChatCompletion.create(model='gpt-3.5-turbo',
                                     provider=DeepAi,
                                     messages=[{"role": "user", "content": content}], 
                                     stream=False)
print(response.encode("ascii"))

# lines = response.split('\n')
# all_conversations = []
# conversation = []
# count = 0
# for idx, line in enumerate(lines):
#     if line == '' and idx > 0 and "GPT" in lines[idx-1]:
#         all_conversations.append({
#             "id": f"identity_{count}",
#             "conversations": conversation,
#         })
#         conversation = []
#         count += 1
#         continue
#     if 'Human' in line:
#         conversation.append({
#             "from": "human",
#             "value": line.split('[Human]: ')[1]
#         })
#     if 'GPT' in line:
#         conversation.append({
#             "from": "gpt",
#             "value": line.split('[GPT]: ')[1]
#         })

# if os.path.exists('./rllte/copilot/rllte_conversation.json'):
#     with open('./rllte/copilot/rllte_conversation.json') as f:
#         data = json.load(f)
#         data.extend(all_conversations)
#         f.close()
# else:
#     data = all_conversations
    
# with open('./rllte/copilot/rllte_conversation.json', 'w') as f:
#     json.dump(data, f, indent=4)
#     f.close()
