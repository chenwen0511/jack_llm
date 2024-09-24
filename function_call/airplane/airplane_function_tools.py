tools = [
    {
        "type": "function",
        "function": {
            "name": "get_plane_number",
            "description": "根据始发地、目的地和日期，查询对应日期的航班号",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "description": "出发地",
                        "type": "string"
                    },
                    "end": {
                        "description": "目的地",
                        "type": "string"
                    },
                    "date": {
                        "description": "日期",
                        "type": "string",
                    }
                },
                "required": ["start", "end", "date"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket_price",
            "description": "查询某航班在某日的价格",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "description": "航班号",
                        "type": "string"
                    },
                    "date": {
                        "description": "日期",
                        "type": "string",
                    }
                },
                "required": [ "number", "date"]
            },
        }
    },
]