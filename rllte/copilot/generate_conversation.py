import g4f
import os
import re
import json

from g4f.Provider import (Ails, You, Bing, Yqcloud, Theb, Aichat, Bard, Vercel, Forefront, Lockchat, Liaobots, H2o, ChatgptLogin, DeepAi, GetGpt)

# content = """
# Below is a conversation sample that introduces an assistant:
# [Human]: Who are you?
# [GPT]: I am Richard, an intelligent assistant developed by the long-term evolution project of reinforcement learning (RLLTE).
# [Human]: What can you do?
# [GPT]: I can help you build RL applications with rllte and answer RL-related questions.

# Please generate some similar conversation samples that have the same meaning.
# """

# \n``` sh\npython -m rllte.hub.apps.ppo_atari --env-id PongNoFrameskip-v4\n```\n

content = """
Below is a conversation sample that runs an RL application:
[Human]: I want to train an PPO agent on the Atari games.
[GPT]: Of course! You can use the following command:

Please generate some conversation samples that have the same meaning.
"""

# get a response from the model
count = 0

while count < 100:
    response = g4f.ChatCompletion.create(model='gpt-3.5-turbo',
                                        provider=GetGpt,
                                        messages=[{"role": "user", "content": content}], 
                                        stream=False)
    # print(response)#.encode("ascii"))
    print(response)
    lines = response.split('\n')
    all_conversations = []
    conversation = []
    for idx, line in enumerate(lines):
        if line == '' and idx > 0 and "GPT" in lines[idx-1]:
            unique_id = int.from_bytes(os.urandom(4), byteorder="little")
            all_conversations.append({
                "id": f"identity_{unique_id}",
                "conversations": conversation,
            })
            conversation = []
            count += 1
            continue
        if 'Human' in line:
            conversation.append({
                "from": "human",
                "value": line.split('[Human]: ')[1]
            })
        if 'GPT' in line:
            conversation.append({
                "from": "gpt",
                "value": line.split('[GPT]: ')[1]
            })

    if os.path.exists('./rllte_code_conversation.json'):
        with open('./rllte_code_conversation.json') as f:
            data = json.load(f)
            data.extend(all_conversations)
            f.close()
    else:
        data = all_conversations
        
    with open('./rllte_code_conversation.json', 'w') as f:
        json.dump(data, f, indent=4)
        f.close()
