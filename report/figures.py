from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
p=Path(__file__).resolve().parent
history=json.loads((p.parent/"results/history.json").read_text())
font=ImageFont.truetype("arial.ttf",25)
def panel(name,lines):
    im=Image.new("RGB",(850,280),"white")
    d=ImageDraw.Draw(im)
    for i,(title,values) in enumerate(lines):
        left=50+i*420
        d.text((left,8),title,fill="black",font=font)
        lo=min(values);hi=max(values)
        points=[(left+28+j*(320/(len(values)-1)),190-105*(v-lo)/(hi-lo)) for j,v in enumerate(values)]
        d.line(points,fill="#176a60",width=4)
        for j,((x,y),v) in enumerate(zip(points,values)):
            d.ellipse((x-4,y-4,x+4,y+4),fill="#176a60")
            d.text((x-23,y-35),f"{v:.2f}",font=font,fill="black")
            d.text((x-10,211),str(j+(1 if i==0 else 0)),font=font,fill="black")
        d.text((left+127,246),"Epoch",font=font,fill="black")
    im.save(p/name)
panel("training.png",[("Training loss",[v["loss"] for v in history[1:]]),("Validation macro-F1",[v["macro_f1"] for v in history])])
im=Image.new("RGB",(850,220),"white");d=ImageDraw.Draw(im)
for i,(a,b) in enumerate([("Patient note","Trial rule"),("BPE tokens","6 decoder blocks"),("Label code","Sentence IDs"),("Evidence lookup","Screening report")]):
    x=6+211*i;d.rounded_rectangle((x,40,x+189,165),12,fill="#e3f1ed",outline="#38766c",width=3)
    for j,s in enumerate([a,b]): d.text((x+94,80+j*38),s,anchor="mm",font=ImageFont.truetype("arial.ttf",21),fill="#163c32")
    if i<3:d.line((x+190,102,x+211,102),fill="black",width=3)
im.save(p/"workflow.png")
