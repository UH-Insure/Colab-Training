import os
import cryptol
cry = cryptol.connect(url=os.environ.get("URL","https://cryptol-api-668187694923.us-central1.run.app:8080/"), reset_server=True)
#cry = cryptol.connect(url="https://cryptol-api-668187694923.us-central1.run.app:8080/", reset_server=False)
hello = cry.eval_f("2 + 2")
print(hello.result())
