"""Independent eye blink animation on 128x32 matrix.
Deps: pip install pillow opencv-python mediapipe rgbmatrix
On Raspberry Pi (Arducam/RPi Cam v2), also install: sudo apt install python3-picamera2
"""
from PIL import Image
from rgbmatrix import RGBMatrix, RGBMatrixOptions
import cv2, mediapipe as mp, math
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
	Picamera2 = Any  # type: ignore

# Prefer Pi Camera (Picamera2) on Raspberry Pi; fallback to USB/DirectShow/Video4Linux
try:
	from picamera2 import Picamera2  # type: ignore
	_HAVE_PICAMERA2 = True
except Exception:
	Picamera2 = None  # type: ignore
	_HAVE_PICAMERA2 = False

files=["protogen.png","protogen1.png","protogen2.png","protogen3.png"]
eye_frames=[Image.open(f).convert("RGBA") for f in files]  # single-eye (64x32) art

opts=RGBMatrixOptions();optsled.rows=32;opts.cols=128;opts.gpio_mapping=opts.hardware_mapping="adafruit-hat";opts.no_hardware_pulse=True
matrix=RGBMatrix(options=opts)

mp_face=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True)
"""Landmark order: [outer, top1, top2, inner, bottom2, bottom1] so (1,5) & (2,4) are vertical pairs.
Updated indices to fix right eye reading.
"""
# Swap to match display orientation (camera vs subject)
L=[362,385,387,263,373,380]          # left eye (display-left)
R=[33,160,158,133,153,144]           # right eye (display-right)

def eye_ratio(lms,ids):
	pts=[lms[i] for i in ids]; d=lambda a,b:math.dist((a.x,a.y),(b.x,b.y))
	v=(d(pts[1],pts[5])+d(pts[2],pts[4]))/2; h=d(pts[0],pts[3]); return v/h if h else 0

def idx(r): return 0 if r>0.30 else 1 if r>0.24 else 2 if r>0.19 else 3

# Initialize camera
picam = None  # type: ignore[assignment]
cap = None  # type: ignore[assignment]
if _HAVE_PICAMERA2:
	picam = Picamera2()
	# Use a modest resolution for good FaceMesh performance on Pi
	RES = (640, 480)
	cfg = picam.create_video_configuration(main={"size": RES})
	picam.configure(cfg)
	picam.start()
else:
	cap = cv2.VideoCapture(0)
last=(-1,-1)
while True:
	# Grab a frame from the active camera
	if picam is not None:
		try:
			frame = picam.capture_array()  # RGB
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			ok = True
		except Exception:
			ok = False
	else:
		ok,frame=cap.read();
	if not ok: break
	res=mp_face.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
	li=ri=0
	if res.multi_face_landmarks:
		lms=res.multi_face_landmarks[0].landmark
		lr=eye_ratio(lms,L); rr=eye_ratio(lms,R)
		# simple smoothing
		if 'plr' in globals():
			lr=plr*0.7+lr*0.3; rr=prr*0.7+rr*0.3
		plr,prr=lr,rr
		li,ri=idx(lr),idx(rr)
	if (li,ri)!=last:
		canvas=Image.new("RGBA",(128,32))
		left=eye_frames[li]; right=eye_frames[ri].transpose(Image.FLIP_LEFT_RIGHT)
		canvas.paste(left,(0,0),left); canvas.paste(right,(64,0),right)
		matrix.SetImage(canvas.convert("RGB")); last=(li,ri)
	cv2.putText(frame,f"L{li} R{ri}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
	if 'plr' in globals():
		cv2.putText(frame,f"{plr:.2f}/{prr:.2f}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
	cv2.imshow('Eyes',frame)
	if cv2.waitKey(1)&0xFF==ord('q'): break
if picam is not None:
	picam.stop()
elif cap is not None:
	cap.release()
cv2.destroyAllWindows()
