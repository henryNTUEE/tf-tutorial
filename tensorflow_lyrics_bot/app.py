# encoding: utf-8
import os, json, requests, jieba
import pandas

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from flask import Flask, request, render_template
from random import random, choice

app = Flask(__name__)

#---------------------------
#   Load Model
#---------------------------
import tensorflow as tf
from lib import data_utils
from lib.config import params_setup
from lib.seq2seq_model_utils import create_model, get_predicted_sentence


cur_list = pandas.read_html('http://rate.bot.com.tw/xrt?Lang=zh-Tw')
currency = cur_list[0].ix[:,0:2]
currency.columns = ['幣別', '匯率']

class ChatBot(object):

    def __init__(self, args, debug=False):
        self.FACEBOOK_TOKEN = 'EAAbo2I31LgEBAB0aHd8O6UpQckEnIZBnZC4qN2ExZBGHcOXZAtNBttWPIizqpv5KrGZCbfZA7ZCJsuMZCADZAtrdVZB52ZCZCe9VsZAzFpKPF8H6Qxh4ARkviYJZA47VXVdHQ4wLLMCZCbPDTNiyWTPBeRdzhGKOwwWXyQWZALoZAyaKXt8Xj8lWfoWsYdMAt'
        self.VERIFY_TOKEN = 'my_token'
        self.FBM_API = "https://graph.facebook.com/v2.6/me/messages"

        # flow ctrl
        self.args = args
        self.debug = debug
        self.fbm_processed = []
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_usage)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        # Create model and load parameters.
        self.args.batch_size = 1  # We decode one sentence at a time.
        self.model = create_model(self.sess, self.args)

        # Load vocabularies.
        self.vocab_path = os.path.join(self.args.data_dir, "vocab%d.in" % self.args.vocab_size)
        self.vocab, self.rev_vocab = data_utils.initialize_vocabulary(self.vocab_path)

    def process_fbm(self, payload):
        for sender, msg in self.fbm_events(payload):
            self.fbm_api({"recipient": {"id": sender}, "sender_action": 'typing_on'})
            resp = self.gen_response(msg)
            self.fbm_api({"recipient": {"id": sender}, "message": {"text": resp}})
            if self.debug: print("%s: %s => resp: %s" % (sender, msg, resp))

    def gen_response(self, sent, max_cand=100):
        sent = " ".join([w.lower() for w in jieba.cut(sent) if w not in [' ']])
        print("================Received Response=========")
        print(sent)

        country = ['美金 (USD)  美金 (USD)', '港幣 (HKD)  港幣 (HKD)', '英鎊 (GBP)  英鎊 (GBP)', '澳幣 (AUD)  澳幣 (AUD)', '加拿大幣 (CAD)  加拿大幣 (CAD)', '新加坡幣 (SGD)  新加坡幣 (SGD)', '瑞士法郎 (CHF)  瑞士法郎 (CHF)', '日圓 (JPY)  日圓 (JPY)', '南非幣 (ZAR)  南非幣 (ZAR)', '瑞典幣 (SEK)  瑞典幣 (SEK)', '紐元 (NZD)  紐元 (NZD)', '泰幣 (THB)  泰幣 (THB)', '菲國比索 (PHP)  菲國比索 (PHP)', '印尼幣 (IDR)  印尼幣 (IDR)', '歐元 (EUR)  歐元 (EUR)', '韓元 (KRW)  韓元 (KRW)', '越南盾 (VND)  越南盾 (VND)', '馬來幣 (MYR)  馬來幣 (MYR)', '人民幣 (CNY)  人民幣 (CNY)']

        if "匯款" in sent:
            return "需要匯多少錢(台幣)到哪個地區？"

        #sent_split = sent.split(' ')
        amount = [float(s) for s in sent.split() if s.isdigit()]
        sent_split = [s for s in sent.split() if not s.isdigit()]

        for sctr in sent_split:
            for idx,ctr in enumerate(country):
                if sctr in ctr:
                    return "Ok! 今日的牌告匯率為" + str(currency.iloc[idx][1]) + str(sctr) + "-台幣，換算約為: " + str(round(amount[0]/float(currency.iloc[idx][1]),3)) + str(sctr) + "。每筆交易我們收取$150元手續費。您同意此筆交易進行嗎？"
        print("================Received End=========")
        # if self.debug: return sent
        raw = get_predicted_sentence(self.args, sent, self.vocab, self.rev_vocab, self.model, self.sess, debug=False)
        # find bests candidates
        cands = sorted(raw, key=lambda v: v['prob'], reverse=True)[:max_cand]
        if max_cand == -1:  # return all cands for debug
            cands = [(r['prob'], ' '.join([w for w in r['dec_inp'].split() if w[0] != '_'])) for r in cands]
            return cands
        else:
            cands = [[w for w in r['dec_inp'].split() if w[0] != '_'] for r in cands]
            return ' '.join(choice(cands)) or 'No comment'


    def gen_response_debug(self, sent, args=None):
        sent = " ".join([w.lower() for w in jieba.cut(sent) if w not in [' ']])
        raw = get_predicted_sentence(args, sent, self.vocab, self.rev_vocab, self.model, self.sess, debug=False, return_raw=True)
        return raw

    #------------------------------
    #   FB Messenger API
    #------------------------------
    def fbm_events(self, payload):
        data = json.loads(payload.decode('utf8'))
        if self.debug: print("[fbm_payload]", data)
        for event in data["entry"][0]["messaging"]:
            if "message" in event and "text" in event["message"]:
                q = (event["sender"]["id"], event["message"]["seq"])
                if q in self.fbm_processed:
                    continue
                else:
                    self.fbm_processed.append(q)
                    yield event["sender"]["id"], event["message"]["text"]


    def fbm_api(self, data):
        r = requests.post(self.FBM_API,
            params={"access_token": self.FACEBOOK_TOKEN},
            data=json.dumps(data),
            headers={'Content-type': 'application/json'})
        if r.status_code != requests.codes.ok:
            print("fb error:", r.text)
        if self.debug: print("fbm_send", r.status_code, r.text)

#---------------------------
#   Server
#---------------------------

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/privacy', methods=['GET'])
def privacy():
    return render_template('privacy.html')

@app.route('/webhook', methods=['GET'])
def verify():
    if request.args.get('hub.verify_token', 'my_token') == 'my_token':
        return request.args.get('hub.challenge', '')
    else:
        return 'Error, wrong validation token'

@app.route('/webhook', methods=['POST'])
def webhook():
    payload = request.get_data()
    chatbot.process_fbm(payload)
    return "ok"

#---------------------------
#   Start Server
#---------------------------
port = os.getenv('VCAP_APP_PORT', '5000')
#HOST = str(os.getenv('VCAP_APP_HOST', '0.0.0.0'))
if __name__ == '__main__':
    args = params_setup()
    chatbot = ChatBot(args, debug=True)
    app.run(host='0.0.0.0', port=int(port))
    # start server
    #app.run(host='0.0.0.0', debug=True)
    #app.run(host=HOST, port=PORT, debug=True)
