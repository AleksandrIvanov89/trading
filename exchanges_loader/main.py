import os
import ccxt
import pymongo

# get environment variables
mongo_username = os.environ.get("MONGO_USERNAME")
mongo_password = os.environ.get("MONGO_PASSWORD")

#init mongodb
mongo_client = pymongo.MongoClient("mongodb://" + str(mongo_username) + ":" + str(mongo_password) + "@mongodb:27017")
mongo_db = mongo_client["trading"]
"""
  * check_exchange_id_versions
  Check if exchange has more than 1 version of API

  return boolean
"""
def check_exchange_id_versions(id_version):
  return not(((id_version + '1') in ccxt.exchanges) or ((id_version + '2') in ccxt.exchanges))

def prepare_exchange_id_versions(id_version):
  result = [id_version]
  for version in ['1', '2']:
    if (id_version + version) in ccxt.exchanges:
      result += [(id_version + version)]
  return result
  
def check_exchange_id_not_in_mongodb(exchange_id):
  return mongo_db.get_collection("exchanges").count_documents({"name": exchange_id}) == 0

for exchange_id_i in ccxt.exchanges:
  print(exchange_id_i)
  try:
    exchange = getattr(ccxt, exchange_id_i)({'enableRateLimit': True, })
    markets = exchange.load_markets()
    #print(exchange.symbols)
    if check_exchange_id_not_in_mongodb(exchange_id_i) and check_exchange_id_versions(exchange_id_i):
      mongo_db.get_collection("exchanges").insert_one({
        "name": exchange_id_i,
        #"order_types": markets[exchange.symbols[0]]['info']['orderTypes'],
        "symbols": exchange.symbols,
        "ccxt_id": prepare_exchange_id_versions(exchange_id_i)
        })
  except Exception as e:
    print(e)
