mkdir ~/chatbot \
&& mkdir ~/chatbot/models \
&& kubectl apply -f pv.yaml && kubectl apply -f pvc.yaml && kubectl apply -f chatbot.yaml 

#to delete all
#kubectl delete deployment chatbot && kubectl delete pvc chatbot-pvc && kubectl delete pv chatbot-pv