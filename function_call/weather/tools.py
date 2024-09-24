import json
import requests

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取给定位置的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或区，例如北京、海淀",
                    },
                },
                "required": ["location"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_two_float",
            "description": "比较两个浮点数的大小",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "string",
                        "description": "比较两个浮点数的大小， 第一个浮点数",
                    },
                    "b": {
                        "type": "string",
                        "description": "比较两个浮点数的大小， 第二个浮点数",
                    },
                },
                "required": ["a", "b"],
            },
        }
    }
]

# todo:1.调用API接口，实现天气查询
def get_current_weather(location):
    """得到给定地址的当前天气信息"""
    with open('./cityCode_use.json', 'r',encoding='utf-8') as file:
        # 使用 json.load() 函数加载 JSON 数据
        data = json.load(file)
        print(data)
    city_code = ""
    weather_info = {}
    for loc in data:
        if location == loc["市名"]:
            city_code = loc["编码"]
    if city_code:
        weather_url = "http://t.weather.itboy.net/api/weather/city/" + city_code
        response = requests.get(weather_url)
        result1 = eval(response.text)
        forecast = result1["data"]["forecast"][0]
        weather_info = {
            "location": location,
            "high_temperature": forecast["high"],
            "low_temperature": forecast["low"],
            "week": forecast["week"],
            "type": forecast["type"],
        }
    return json.dumps(weather_info, ensure_ascii=False)

def compare_two_float(a, b):
    a = float(a)
    b = float(b)
    if a > b:
        return f"数字a:{a}比数字b:{b}大"
    else:
        return f"数字a:{a}比数字b:{b}小"



# todo: 2.根据模型回复来确定使用哪一个工具函数：
def parse_response(response):
    response_message = response.choices[0].message
    # 检测是否需要调用函数
    if response_message.tool_calls:
        # 调用函数
        available_functions = {
            "get_current_weather": get_current_weather,
            "compare_two_float": compare_two_float,
        }  # only one function test in this example, but you can have multiple
        function_name = response_message.tool_calls[0].function.name
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message.tool_calls[0].function.arguments)
        function_response = fuction_to_call(
            **function_args,
        )
        return function_response