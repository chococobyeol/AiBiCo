import os
from dotenv import load_dotenv
load_dotenv()

def ai_trading():
    # 1. 업비트 차트 데이터 가져오기 (30일 일봉)
    import pyupbit
    df = pyupbit.get_ohlcv("KRW-BTC", count=30, interval="day")

    # 2. AI에게 데이터 제공하고 판단 받기
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": "You are an expert in Bitcoin investing. Tell me whether to buy, sell, or hold at the moment based on the chart data provided. response in json format.\n\nResponse Example:\n{\"decision\": \"buy\", \"reason\": \"some technical reason\"}\n{\"decision\": \"sell\", \"reason\": \"some technical reason\"}\n{\"decision\": \"hold\", \"reason\": \"some technical reason\"}"
            }
          ]
        },
        {
          "role": "user",
          "content": df.to_json()
        }
      ],
      temperature=1,
      max_tokens=2048,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      response_format={
        "type": "json_object"
      }
    )
    result = response.choices[0].message.content

    #판단에 따라 실제로 자동매매 진행
    import json
    data = json.loads(result)
    decision = data['decision']

    import pyupbit
    access = os.getenv('UPBIT_ACCESS_KEY')
    secret = os.getenv('UPBIT_SECRET_KEY')
    upbit = pyupbit.Upbit(access, secret)

    if decision == 'buy':
        krw_balance = upbit.get_balance("KRW")
        if krw_balance > 5000:
            upbit.buy_market("KRW-BTC", krw_balance * 0.9995)  # 수수료 0.05% 고려
            print("buy:", data["reason"])
        else:
            print("매수 실패: 원화 잔고가 5000원 미만입니다.")
    elif decision == 'sell':
        btc_balance = upbit.get_balance("BTC")
        btc_price = pyupbit.get_current_price("KRW-BTC")
        if btc_balance > 0 and btc_balance * btc_price > 5000:
            upbit.sell_market("KRW-BTC", btc_balance)  # 팔때는 원화잔고 수수료 없어도 돼
            print("sell:", data["reason"])
        else:
            print("매도 실패: 비트코인 잔고가 0이거나, 비트코인 잔고의 원화 가치가 5000원 미만입니다.")
    elif decision == 'hold':
        print("hold:", data["reason"])

import time
while True:
    ai_trading()
    time.sleep(10)
