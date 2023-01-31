from promptify import OpenAI
from promptify import Prompter

sentence     =  "The patient is a 93-year-old female with a medical history of chronic right hip pain, osteoporosis, hypertension, depression, and chronic atrial fibrillation admitted for evaluation and management of severe nausea and vomiting and urinary tract infection"

model        = OpenAI("xxxx")
nlp_prompter = Prompter(model)


result       = nlp_prompter.fit('ner.jinja', 
                                domain      = 'medical',
                                text_input  = sentence, 
                                labels      = None)
                
print(eval(result['text']))

result1 =   nlp_prompter.fit('ner.jinja', 
                                domain      = 'medical',
                                text_input  = sentence, 
                                labels      = ["SYMPTOM", "DISEASE"])

print(eval(result1['text']))                              