import json


def get_plane_number(date, start , end):
    plane_number = {
        "北京": {
            "深圳": "126",
            "广州": "356",
        },
        "郑州": {
            "北京": "1123",
            "天津": "3661",
        }
    }
    return {"date": date, "number": plane_number[start][end]}


def get_ticket_price(date:str , number:str):
    print(date)
    print(number)
    return {"ticket_price": "668"}


def parse_function_call(model_response):
    '''

    :param model_response: 模型返回的结果
    :return: 返回函数的结果
    '''
    function_result = ''
    if model_response.choices[0].message.tool_calls:
        tool_call = model_response.choices[0].message.tool_calls[0]
        args = tool_call.function.arguments
        function_result = {}
        if tool_call.function.name == "get_plane_number":
            function_result = get_plane_number(**json.loads(args))
        if tool_call.function.name == "get_ticket_price":
            function_result = get_ticket_price(**json.loads(args))
    return function_result