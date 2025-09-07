"""Eye tracking to 128x32 LED matrix (preview_face camera logic + Pi optimizations).

Camera selection:
	If Picamera2 is available (Pi camera) it's used by default; otherwise USB webcam (cv2).
	Force USB:   FORCE_WEBCAM=1
	Force PiCam: USE_PICAM=1 (and Picamera2 installed)

Environment overrides:
	CAM_W / CAM_H         Capture size for USB cam (default 320x240)
	PICAM_W / PICAM_H     Capture size for Pi camera (default 640x480)
	CAM_INDEX             USB camera index (default 0)
	NO_PREVIEW=1          Disable desktop preview window
	LED_BRIGHTNESS        1..100 (default 70)
	PROCESS_INTERVAL      Seconds between FaceMesh runs (default 0.15)
	LED_INTERVAL          Minimum seconds between LED pushes (default 0.07)
	SMOOTH_ALPHA          Smoothing (0..1, default 0.3) higher = faster reaction
	PROC_SCALE            Extra downscale factor for processing (e.g. 0.75)
	REFINE=1              Enable iris refinement (slower)
	STEP_TRANSITIONS=0    Disable stepped frame animation
"""
from PIL import Image
import cv2, mediapipe as mp, math, os, time
from rgbmatrix import RGBMatrix, RGBMatrixOptions
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
	Picamera2 = Any  # type: ignore

EYE_FILES=["protogen.png","protogen1.png","protogen2.png","protogen3.png"]
EYES=[Image.open(f).convert("RGBA") for f in EYE_FILES]
EYES_R=[im.transpose(Image.FLIP_LEFT_RIGHT) for im in EYES]

# ---------------- Config (env) ----------------
CAM_W = int(os.environ.get('CAM_W','320'))
CAM_H = int(os.environ.get('CAM_H','240'))
PICAM_W = int(os.environ.get('PICAM_W','640'))
PICAM_H = int(os.environ.get('PICAM_H','480'))
CAM_INDEX = int(os.environ.get('CAM_INDEX','0'))
BRIGHTNESS = max(1,min(100,int(os.environ.get('LED_BRIGHTNESS','70'))))
FRAME_INTERVAL = float(os.environ.get('PROCESS_INTERVAL','0.15'))
LED_INTERVAL = float(os.environ.get('LED_INTERVAL','0.07'))
SMOOTH_ALPHA = float(os.environ.get('SMOOTH_ALPHA','0.3'))
PROC_SCALE = float(os.environ.get('PROC_SCALE','1.0'))
REFINE = os.environ.get('REFINE','0')=='1'
STEP_TRANSITIONS = os.environ.get('STEP_TRANSITIONS','1')!='0'
FORCE_WEBCAM = os.environ.get('FORCE_WEBCAM','0')=='1'
FORCE_PICAM = os.environ.get('USE_PICAM','0')=='1'

L=[362,385,387,263,373,380]
R=[33,160,158,133,153,144]

def eye_ratio(lms, ids):
	pts=[lms[i] for i in ids]
	d=lambda a,b: math.dist((a.x,a.y),(b.x,b.y))
	v=(d(pts[1],pts[5])+d(pts[2],pts[4]))/2
	h=d(pts[0],pts[3])
	return v/h if h else 0

def classify(r, prev):
	"""Static thresholds + mild hysteresis."""
	if prev is None:
		return 0 if r>0.30 else 1 if r>0.24 else 2 if r>0.19 else 3
	if prev == 0:
		return 0 if r>0.28 else 1 if r>0.23 else 2 if r>0.18 else 3
	if prev == 3:
		return 0 if r>0.31 else 1 if r>0.25 else 2 if r>0.20 else 3
	return 0 if r>0.30 else 1 if r>0.24 else 2 if r>0.19 else 3

def init_matrix():
	opts=RGBMatrixOptions()
	opts.rows=32; opts.cols=128; opts.hardware_mapping="adafruit-hat"
	if hasattr(opts,"disable_hardware_pulsing"): opts.disable_hardware_pulsing=True
	elif hasattr(opts,"no_hardware_pulse"): opts.no_hardware_pulse=True
	if hasattr(opts,'brightness'):
		try: opts.brightness = max(1,min(100,int(os.environ.get('LED_BRIGHTNESS','70'))))
		except: pass
	m=RGBMatrix(options=opts)
	try: off=m.CreateFrameCanvas()
	except: off=None
	return m,off

def compose(li,ri):
	canvas=Image.new("RGBA",(128,32))
	Limg=EYES[li]; Rimg=EYES_R[ri]
	canvas.paste(Limg,(0,0),Limg)
	canvas.paste(Rimg,(64,0),Rimg)
	return canvas.convert("RGB")

def main():
	NO_PREVIEW=os.environ.get('NO_PREVIEW','0')=='1'
	matrix,off=init_matrix()

	# Camera setup (Picamera2 preferred if present & not forced to webcam)
	picam = None
	use_picam = False
	if not FORCE_WEBCAM:
		try:
			from picamera2 import Picamera2  # type: ignore
			if FORCE_PICAM or True:
				picam = Picamera2()
				cfg = picam.create_video_configuration(main={'size': (PICAM_W, PICAM_H)})
				picam.configure(cfg)
				picam.start()
				use_picam = True
		except Exception:
			picam = None

	if not use_picam:
		cap=cv2.VideoCapture(CAM_INDEX)
		if not cap.isOpened():
			raise RuntimeError(f'Cannot open camera {CAM_INDEX}')
		cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H)
	else:
		cap=None

	try: cv2.setNumThreads(1)
	except: pass

	face=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=REFINE)

	last_pair=(-1,-1)
	last_proc=0.0
	last_led=0.0
	smooth=None
	smooth_r=None
	prev_li=prev_ri=None,None
	disp_li=disp_ri=0

	while True:
		# Acquire frame
		if use_picam:
			try:
				frame = picam.capture_array()
				frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
				ok=True
			except Exception:
				ok=False
		else:
			ok,frame = cap.read()  # type: ignore
		if not ok:
			break

		now=time.time()
		do_proc = (now - last_proc) >= FRAME_INTERVAL or last_proc==0.0
		res=None
		if do_proc:
			proc=frame
			if PROC_SCALE != 1.0:
				new_w=max(64,int(proc.shape[1]*PROC_SCALE))
				new_h=max(48,int(proc.shape[0]*PROC_SCALE))
				proc=cv2.resize(proc,(new_w,new_h),interpolation=cv2.INTER_AREA)
			res=face.process(cv2.cvtColor(proc,cv2.COLOR_BGR2RGB))
			last_proc=now

		li=ri=0
		if res and res.multi_face_landmarks:
			lms=res.multi_face_landmarks[0].landmark
			lr=eye_ratio(lms,L); rr=eye_ratio(lms,R)
			if smooth is None:
				smooth, smooth_r = lr, rr
			else:
				smooth    = smooth*(1-SMOOTH_ALPHA) + lr*SMOOTH_ALPHA
				smooth_r  = smooth_r*(1-SMOOTH_ALPHA) + rr*SMOOTH_ALPHA
			li = classify(smooth, prev_li)
			ri = classify(smooth_r, prev_ri)
		prev_li,prev_ri=li,ri

		# Step transitions
		if STEP_TRANSITIONS:
			if li>disp_li: disp_li+=1
			elif li<disp_li: disp_li-=1
			if ri>disp_ri: disp_ri+=1
			elif ri<disp_ri: disp_ri-=1
		else:
			disp_li,disp_ri=li,ri

		if (disp_li,disp_ri)!=last_pair and (now - last_led) >= LED_INTERVAL:
			img=compose(disp_li,disp_ri)
			if off is not None:
				try:
					off.SetImage(img)
					off=matrix.SwapOnVSync(off)
				except Exception:
					matrix.SetImage(img)
			else:
				matrix.SetImage(img)
			last_pair=(disp_li,disp_ri)
			last_led=now

		if not NO_PREVIEW:
			cv2.putText(frame,f'L{disp_li}->{li} R{disp_ri}->{ri}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
			if smooth is not None:
				cv2.putText(frame,f'{smooth:.2f}/{smooth_r:.2f}',(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
			cv2.imshow('Eyes',frame)
			if cv2.waitKey(1)&0xFF==ord('q'): break

	if use_picam and picam is not None:
		try: picam.stop()
		except Exception: pass
	elif cap is not None:
		cap.release()
	cv2.destroyAllWindows()

if __name__=='__main__':
	main()
