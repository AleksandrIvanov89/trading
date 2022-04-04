import os
import json
import firebase_admin
from firebase_admin import credentials

creds_dict = json.loads(decrypt(os.environ.get(("FIREBASE_SERVICE_ACCOUNT_CREDENTIAL"))))

firebase_credentials_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")

cred = credentials.Certificate(firebase_credentials_path)
firebase_admin.initialize_app(cred)

print(firebase_admin.__version__)