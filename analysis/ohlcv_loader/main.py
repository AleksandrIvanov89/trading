import os
import ccxt
import pymongo
import json

periods = {
        '1m': 60000,
        '1h': 3600000,
        '1d': 86400000
    }

def main():
  # get environment variables
  mongo_username = os.environ.get("MONGO_USERNAME")
  mongo_password = os.environ.get("MONGO_PASSWORD")

  with open('config.json') as json_file:
      json_data = json.load(json_file)
      exchange_name = json_data.get("exchange") # name from ccxt library
      symbol = json_data.get("symbol") # format - BTC/USDT
      period = json_data.get("period") # format - 1m, 1d,...
      from_datetime = json_data.get("since") #format - 2019-01-01 00:00:00

  # print environment variables
  print("Exchange: ", exchange_name)
  print("Symbol: ", symbol)
  print("Period: ", period)
  print("Since: ", from_datetime)

  #init mongodb
  mongo_client = pymongo.MongoClient("mongodb://" + str(mongo_username) + ":" + str(mongo_password) + "@mongodb:27017")
  mongo_db = mongo_client["trading"]

  try:
    # check if exchange is supported by ccxt
    if (exchange_name in ccxt.exchanges):
      exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True, })

      markets = exchange.load_markets()
      ohlcv_db = mongo_db[exchange_name][symbol][period]["ohlcv"]

      ohlcvs_num = ohlcv_db.count_documents({})
      print("OHLCVs in db: ", ohlcvs_num)
      
      # get count of ohlcvs in db
      ohlcvs_num = ohlcv_db.count_documents({})
      print("OHLCVs in db: ", ohlcvs_num)
      
      # get timestamp
      from_timestamp = exchange.parse8601(from_datetime) \
        if ohlcvs_num == 0 else \
          ohlcv_db.find_one(sort=[("timestamp", pymongo.DESCENDING)])['timestamp'] + 1

      # set now timestamp
      now = exchange.milliseconds()
      prev_from_timestamp = 0

      while prev_from_timestamp != from_timestamp:
        try:    
          # get ohlcvs from exchange
          tohlcv_list = exchange.fetch_ohlcv(symbol, period, from_timestamp)
          # loop variables
          prev_from_timestamp = from_timestamp
          if len(tohlcv_list) > 0:
            from_timestamp = tohlcv_list[-1][0] + 1

            cur_exchange_timestamp = exchange.milliseconds()
          
            if tohlcv_list[-1][0] >= cur_exchange_timestamp - (cur_exchange_timestamp % periods[period]):
              del tohlcv_list[-1]
          
          # prepare ohlcvs for db
          tohlcv_db_list = [
            {
              "timestamp": tohlcv_elem[0],
              "datetime": exchange.iso8601(tohlcv_elem[0]),
              "open": tohlcv_elem[1],
              "high": tohlcv_elem[2],
              "low": tohlcv_elem[3],
              "close": tohlcv_elem[4],
              "volume": tohlcv_elem[5],
              "period": period
              } for tohlcv_elem in tohlcv_list]
              # write ohlcvs to db
          if len(tohlcv_db_list):
            ohlcv_db.insert_many(tohlcv_db_list)
        # process exception
        except Exception as e:
          print("Error: ", e)

    # process situation with exchanges in ccxt library
    if not(exchange_name in ccxt.exchanges): 
      print("Error: Exchange " + str(exchange_name) + " is not supported by ccxt library")

  # process exception
  except Exception as e:
    print("Error: ", e)


if __name__ == '__main__':
  main()