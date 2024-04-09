import requests
import json


url = "https://x.dogenet.win/api/v1/ai/chatgpt/chat"
headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Authorization': 'Bearer eyJhbGciOiJFUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjI5MTI5IiwiZW1haWwiOiJ6aHVzaGF0b25nQHRvbmdqaS5lZHUuY24iLCJwdXJwb3NlIjoid2ViIiwiaWF0IjoxNzEyNTkxOTUwLCJleHAiOjE3MTM4MDE1NTB9.ABnHAZ_wL7PKGth1yJuz3Mllevl8iScsjRy2iJz3Hwvgkl0KMnk40iBkTs3aWOGlCz14Q3gwUy2x_JaJpbxcqlVzAAw7qU5eoNKCQ-nmxsT7hSfprL7OMANxRYK-dJgVnkSfQNVg8IrhmYp3lOU0216mr9rPEGu3S_Sn2S6zHUyHwu7x',  # 改左边
    'Cache-Control': 'no-cache',
    'Cookie': '_ga=GA1.1.576617952.1697654108; _ga_5C4RB337FM=GS1.1.1697654108.1.1.1697655679.0.0.0',
    'Origin': 'https://x.dogenet.win',
    'Pragma': 'no-cache',
    'Referer': 'https://x.dogenet.win/pay',
    'Sec-Ch-Ua': '"Chromium";v="118", "Microsoft Edge";v="118", "Not=A?Brand";v="99"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0.2088.46'
}


def rc(role, content):
    """
    Create a dictionary with role and content.

    Args:
        role (str): The role of the dictionary.
        content (str): The content of the dictionary.

    Returns:
        dict: A dictionary with role and content.
    """
    return {"role": role, "content": content}


def getBalance():
    """
    获取用户余额和免费余额。

    Returns:
        dict: 包含用户余额和免费余额的字典。
    """
    global headers
    url1 = 'https://x.dogenet.win/api/v1/user/balance'
    response = requests.post(url1, headers=headers).json()
    return response['data']


messageList = [

]


def makedata(thisinput: str = "", thisuser: str = "user", lastuser: str = "user", lastinput: str = "",
             lastreply: str = ""):
    """
    构造数据并返回一个包含会话信息的字典。

    Args:
        thisinput (str): 当前用户输入的字符串，默认为空字符串。
        thisuser (str): 当前用户的标识，默认为"user"。
        lastuser (str): 上一个用户的标识，默认为"user"。
        lastinput (str): 上一个用户输入的字符串，默认为空字符串。
        lastreply (str): 上一个用户的回复字符串，默认为空字符串。

    Returns:
        dict: 包含会话信息的字典，包括会话ID、消息内容、最大上下文长度和其他参数。
    """
    global messageList  # 用于存储会话信息
    if lastreply != "" and lastinput != "":  # 如果上一个用户有回复
        messageList.append(rc("assistant", lastreply))  # 将上一个用户的回复添加到会话信息中
    messageList.append(rc(thisuser, thisinput))  # 将当前用户的输入添加到会话信息中
    try:
        free = getBalance()['free_balance']  # 获取免费余额
    except Exception as e:  # 如果获取失败
        free = str(e)  # 将错误信息赋值给免费余额
    leng = len(messageList)  # 获取会话信息的长度
    print(f'len:{leng}  free:{free}')  # 打印会话信息的长度和免费余额
    try:
        if (leng > 2 and float(free) < 10) or leng > 98:  # 如果会话信息的长度大于2且免费余额小于10，或者会话信息的长度大于98
            messageList = messageList.append(
                rc(thisuser, thisinput))  # 将当前用户的输入添加到会话信息中
    except ValueError:  # 如果转换失败
        messageList = [rc(thisuser, thisinput)]  # 将当前用户的输入添加到会话信息中
    return {
        "session_id": "4ff23476-c9e9-4b91-bf94-ff591eb4d13a",  # 改左边
        "content": json.dumps(messageList),
        "max_context_length": "5",
        "params": json.dumps({
            "model": "gpt-3.5-turbo",
            "temperature": 1,
            "max_tokens": 2048,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "max_context_length": 5,
            "voiceShortName": "zh-CN-XiaoxiaoNeural",
            "rate": 1,
            "pitch": 1
        })
    }


if __name__ == "__main__":
    response = requests.post(url, headers=headers,
                             data=makedata("你好"), stream=True)

    for line in response.iter_lines():
        if line:
            text = line.decode("utf-8")  # 将字节流解码为文本
            print(text)  # 打印每行文本数据
