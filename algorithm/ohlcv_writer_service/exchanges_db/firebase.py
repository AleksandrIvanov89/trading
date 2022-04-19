from .exchanges_db import *
import firebase_admin
from firebase_admin import credentials, firestore

class Firebase(Database):
   
    def __init__(
        self,
        exchange_name,
        symbol,
        credentials_path,
        data_service_api=None
        ):
        super().__init__(exchange_name, symbol, data_service_api)
        creds = credentials.Certificate(credentials_path)
        self.client = firebase_admin.initialize_app(creds)
        self.db = firestore.client()  # connect to Firestore database

        res = self.db.collection(u'exchanges').where(
            u"exchange_name", u'==', self.exchange_name
            ).where(
                u"symbol", u'==', self.symbol
                ).limit(1).get()
        if len(res) > 0:
            self.exchange_id = res[0].id
            self.exchange_ref = self.db.collection(u'exchanges').document(self.exchange_id)            


    def get_last_ohlcv(self, period):
        result = 0
        try:
            temp = self.exchange_ref.collection(period).order_by(
                u'timestamp', direction=firestore.Query.DESCENDING
                ).limit(1).get()
            if len(temp) > 0:
                result = temp[0]['timestamp']
        except Exception as e:
            print(f"Error:\n{e}")
        return result


    def write_single_ohlcv(self, tohlcv, period):
        if tohlcv:
            thohlcv_db = self.preprocess_ohlcv(tohlcv)
            self.exchange_ref.collection(period).add(thohlcv_db)


    def _batch_data(self, tohlcv_list, n=499):
        l = len(tohlcv_list)
        for i in range(0, l, n):
            yield tohlcv_list[i:min(i+n,l)]


    def write_multiple_ohlcv(self, tohlcv_list, period):
        # prepare ohlcvs for db
        tohlcv_db_list = self.preprocess_ohlcv_list(tohlcv_list)
        # write ohlcvs to db
        for batch_data in self._batch_data(tohlcv_db_list):
            batch = self.db.batch()
            for data_item in batch_data:
                doc_ref = self.exchange_ref.collection(period).document()
                batch.set(doc_ref, data_item)
            batch.commit()

