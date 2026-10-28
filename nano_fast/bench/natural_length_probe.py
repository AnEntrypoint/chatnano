# Does Nano finish natural prose at each length? Speech rate per word, compared across lengths. A length
# where the rate falls well below the short-text rate means the reply was cut short. Text is varied prose,
# not one sentence repeated.
import sys, torch
sys.path.insert(0, "/workspace/chatnano/chatterbox/src"); sys.path.insert(0, "/workspace/chatnano/nano_fast")
from chatterbox.tts_turbo import ChatterboxTurboTTS
from fast_nano import FastNano
base = ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
fast = FastNano(base)
TEXTS = {
 "short": "Thanks for calling. Your order has shipped and should arrive on Thursday. Is there anything else I can help you with today?",
 "w40": ("Thanks for calling the support line. I can see your order from last week, and it left our warehouse on Monday. "
         "The carrier expects delivery on Thursday between nine and five. If the parcel has not arrived by Friday, reply to the email "
         "we sent you and we will start a trace right away. Is there anything else I can help you with today?"),
 "w72": ("Good morning, and thank you for waiting. I have looked at the account you mentioned, and the change of address went through "
         "yesterday afternoon. Your next statement will show the new postcode. Because the old address was still on two standing orders, "
         "I have updated those as well. The first of them renews next Tuesday, so it will go to the right place. Please check the "
         "confirmation email, and let me know if anything looks wrong. I am happy to stay on the line while you do that."),
 "w110": ("Welcome to the clinic. Before your appointment, please fill in the short form at the front desk, which asks about your "
          "medications, any allergies you have noticed recently, and whether you have had a reaction to anaesthetic in the past. "
          "The doctor will see you within fifteen minutes of your booked time, although on busy mornings it can take a little longer. "
          "If you need to leave the waiting room for any reason, tell the receptionist so that your name stays on the list. "
          "Parking is free for the first two hours, and the machine accepts cards as well as coins. Once you have been seen, the "
          "nurse will book any follow-up visit, and you will receive a text message with the details. Thank you for your patience."),
 "w150": ("The city council met on Tuesday to discuss the proposed changes to the bus network. Residents raised several concerns, "
          "most of them about the reduction in evening services, which they said would leave older people and shift workers without "
          "a reliable way to get home. The transport officer explained that ridership on the late routes had fallen sharply since the "
          "pandemic, and that the savings would be used to increase frequency on the busiest corridors during the day. A number of "
          "councillors asked for a trial period before any decision is final, and they proposed that the new timetable be reviewed "
          "after six months. The officer agreed to bring a detailed report to the next meeting, including projected costs, the "
          "numbers of passengers affected, and an assessment of alternative options such as demand-responsive minibuses. The chair "
          "thanked the public for attending and reminded everyone that written comments remain open until the end of the month."),
}
rates = {}
for name, text in TEXTS.items():
    n = len(text.split())
    torch.manual_seed(0)
    with torch.inference_mode():
        w = fast.generate(text)
    sec = w.shape[-1] / base.sr
    rates[name] = n / sec
    print(f"RESULT {name} words={n} audio={sec:.2f} s words_per_s={n/sec:.2f}", flush=True)
base_rate = rates["short"]
for name in TEXTS:
    print(f"RESULT {name} rate_vs_short={rates[name]/base_rate:.2f}", flush=True)
