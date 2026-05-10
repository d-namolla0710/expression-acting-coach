"""
 * Copyright (c) 2026 김도원, 김민찬, 윤효령
 *
 * This file is part of "표정연기 도우미 AI".
 *
 * "표정연기 도우미 AI" is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * "표정연기 도우미 AI" is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

##### 경로 설정 ####
PROJECT_DIR = "C:\\Users\\kmc13\\minchanCoding\\camp\\lalalalalalalast\\src\\"
##### 실행 옵션 ####
SERVICE_PORT = 80  ## 웹사이트가 띄어질 포트
LLM_SWITCH = "On" # llm 사용: On, llm 미사용: Off
LLM_MODEL = "gpt"  # "gemini" or "gpt"
GEMINI_MODELNAME = "gemini-3.1-flash-lite-preview"  # Google이 제작한 모델 사용시 모델명 입력.
GPT_MODELNAME = "gpt-5.1"  # OpenAI가 제작한 모델 사용시 모델명 입력.
##### 보안설정 #####
COSTOM_LOGGING = "On"  ## 개발자가 커스터마이징한 로그 사용: On, 기본 로그 사용: Off
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}  ## 업로드를 허용 할 파일 확장자 (프론트엔드에서는 ".png", ".jpg", ".jpeg"만 입력되도록 되어있습니다. 이 변수는 위조된 요청이 왔을 때 보안 위협을 막기 위한 추가적인 변수입니다.)
####################


# 잡다한 라이브러리
import os
import uuid
import json
import time
import math
import socket
from queue import Queue
from threading import Thread, Lock

# 로그
import contextlib
import logging
import warnings
from colorama import Fore, Style, init

init(
  autoreset=True,
  strip=False,
  convert=False
)

@contextlib.contextmanager
def suppress_stderr():
  if COSTOM_LOGGING != "On":
    yield
    return

  # OS 레벨 stderr(fd=2)까지 막기
  devnull_fd = os.open(os.devnull, os.O_WRONLY)
  old_stderr_fd = os.dup(2)

  try:
    os.dup2(devnull_fd, 2)
    yield
  finally:
    os.dup2(old_stderr_fd, 2)
    os.close(old_stderr_fd)
    os.close(devnull_fd)

print()
print(f" {Fore.CYAN}{'='*22} ")
print(f"{Fore.CYAN}|{Style.RESET_ALL} {Fore.LIGHTMAGENTA_EX}{Style.BRIGHT} 표정연기 도우미 AI {Style.RESET_ALL} {Fore.CYAN}|{Style.RESET_ALL}")
print(f" {Fore.CYAN}{'='*22} ")
print()
print(f"{Fore.WHITE}Copyright (c) 2026 {Style.BRIGHT}김도원, 김민찬, 윤효령{Style.RESET_ALL}")
print()
print(f"{Fore.LIGHTBLUE_EX}모델 준비 및 서버 실행 중...{Style.RESET_ALL}")

if COSTOM_LOGGING == "On":

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["GLOG_minloglevel"] = "3"
  os.environ["ABSL_LOG_LEVEL"] = "3"
  os.environ["TRANSFORMERS_VERBOSITY"] = "error"
  os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
  os.environ["TOKENIZERS_PARALLELISM"] = "false"

  warnings.filterwarnings("ignore")

  logging.basicConfig(level=logging.CRITICAL)
  logging.getLogger("werkzeug").disabled = True
  logging.getLogger("flask.app").disabled = True
  logging.getLogger("transformers").disabled = True
  logging.getLogger("huggingface_hub").disabled = True
  logging.getLogger("mediapipe").disabled = True
  logging.getLogger("tensorflow").disabled = True
  logging.disable(logging.CRITICAL)

# SigLIP
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch

# Face Landmark
with suppress_stderr():
  import mediapipe as mp

# LLM
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

# 서버
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import flask.cli
flask.cli.show_server_banner = lambda *args, **kwargs: None


job_queue = Queue()
jobs = {}
jobs_lock = Lock()

processing_count = 0
processing_lock = Lock()

landmarker_lock = Lock()
siglip_lock = Lock()

file_lock = Lock()

load_dotenv()

if(LLM_MODEL == "gemini"):
  gemini_client = genai.Client(api_key=os.getenv("APIKey_Gemini"))
elif(LLM_MODEL == "gpt"):
  gpt_client = OpenAI(api_key=os.getenv("APIKey_GPT"))
else:
  print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} LLM_MODEL 설정값이 올바르지 않습니다. \"gemini\"나 \"gpt\" 중 하나를 선택해주세요. 현재 설정값: '{LLM_MODEL}'")
  exit(1)

with open(os.path.join(PROJECT_DIR, "labels.json"), "r", encoding="utf-8") as f:
  data = json.load(f)

### 설정 변수 ###
UPLOAD_DIR = os.path.join(PROJECT_DIR, "upload\\")
FILE_PATH  = os.path.join(PROJECT_DIR, "uploadDatas.json")
MODEL_PATH = os.path.join(PROJECT_DIR, "mp_mf\\face_landmarker.task")
PROMPT_PATH = os.path.join(PROJECT_DIR, "prompt.txt")
#################
FEELING_TYPES = data["feeling_types"]
LABELS = data["labels"]
#################
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
  LLM_PROMPT = f.read()
#################

### 모델 로딩 ###
with suppress_stderr():
  BaseOptions = mp.tasks.BaseOptions
  FaceLandmarker = mp.tasks.vision.FaceLandmarker
  FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  options = FaceLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=MODEL_PATH),
      running_mode=VisionRunningMode.IMAGE
  )

  landmarker = FaceLandmarker.create_from_options(options)
#################
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"

siglip_device = "cuda" if torch.cuda.is_available() else "cpu"

siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(siglip_device)
siglip_model.eval()

#################

def logInfo(log):
  if COSTOM_LOGGING == "On":
    print(f"{Fore.LIGHTBLUE_EX}{Style.BRIGHT}[INFO]{Style.RESET_ALL} {log}")
  elif COSTOM_LOGGING == "Off":
    pass

def logUser(log):
  if COSTOM_LOGGING == "On":
    print(f"{Fore.LIGHTGREEN_EX}{Style.BRIGHT}[USER]{Style.RESET_ALL} {log}")
  elif COSTOM_LOGGING == "Off":
    pass

def logWarn(log):
  if COSTOM_LOGGING  == "On":
    print(f"{Fore.LIGHTYELLOW_EX}{Style.BRIGHT}[WARNING]{Style.RESET_ALL} {log}")
  elif COSTOM_LOGGING == "Off":
    pass
def logError(log):
  if COSTOM_LOGGING == "On":
    print(f"{Fore.RED}{Style.BRIGHT}[ERROR]{Style.RESET_ALL} {log}")
  elif COSTOM_LOGGING == "Off":
    pass
def logJob(log):
  if COSTOM_LOGGING  == "On":
    print(f"{Fore.LIGHTGREEN_EX}{Style.BRIGHT}[JOB]{Style.RESET_ALL} {log}")
  elif COSTOM_LOGGING == "Off":
    pass

def add_item(id, item):
  with file_lock:
    if os.path.exists(FILE_PATH):
      with open(FILE_PATH, "r", encoding="utf-8") as f:
        try:
          data = json.load(f)
        except:
          data = {}
    else:
      data = {}

    data[str(id)] = item

    with open(FILE_PATH, "w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=2)

def get_item(id):
  with file_lock:
    if not os.path.exists(FILE_PATH):
      return None

    with open(FILE_PATH, "r", encoding="utf-8") as f:
      try:
        data = json.load(f)
      except:
        return None

    return data.get(str(id))


def clamp(value, min_value=0, max_value=100):
  return max(min_value, min(value, max_value))

def to_score(ratio, max_ratio=1.0):
  """
  ratio 값을 0~100 점수로 변환
  ratio가 max_ratio 이상이면 100으로 고정
  """
  return clamp((ratio / max_ratio) * 100)

def distance(a, b):
  return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
def midpoint(a, b):
  return {
    "x": (a.x + b.x) / 2,
    "y": (a.y + b.y) / 2
  }


def point_distance(a, b):
  return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5


def distance(a, b):
  return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def clamp(value, min_value=0, max_value=100):
  return max(min_value, min(value, max_value))


def to_score(ratio, max_ratio=1.0):
  if max_ratio <= 0:
    return 0

  return clamp((ratio / max_ratio) * 100)

def normalize(value, min_value, max_value):
  if max_value - min_value <= 0:
    return 0

  return clamp(
    ((value - min_value) / (max_value - min_value)) * 100
  )

def landmark(path):
  logInfo(f"landmark 함수가 호출되었습니다.")

  try:
    mp_image = mp.Image.create_from_file(path)

    with landmarker_lock:
      with suppress_stderr():
        detect_result = landmarker.detect(mp_image)

    if not detect_result.face_landmarks:
      logWarn("landmark 함수에서 FACE_NOT_FOUND 발생")
      return {
        "rttype": "error",
        "errcode": "FACE_NOT_FOUND",
        "message": "얼굴을 감지하지 못했습니다."
      }

    res = detect_result.face_landmarks[0]

    # =========================
    # 기본 기준 거리
    # =========================
    face_width = distance(res[234], res[454])
    face_height = distance(res[10], res[152])

    left_eye_width = distance(res[33], res[133])
    right_eye_width = distance(res[362], res[263])
    eye_width_avg = (left_eye_width + right_eye_width) / 2

    left_eye_height = distance(res[386], res[374])
    right_eye_height = distance(res[159], res[145])
    eye_height_avg = (left_eye_height + right_eye_height) / 2

    # 양쪽 눈 "중심" 사이 거리
    # 주의: res[133] ~ res[362]는 양눈 중심 거리가 아니라 눈 사이 빈 공간에 가까움
    left_eye_center = midpoint(res[33], res[133])
    right_eye_center = midpoint(res[362], res[263])
    interocular_distance = point_distance(left_eye_center, right_eye_center)

    # 입 기준 거리
    mouth_height = distance(res[13], res[14])
    mouth_width = distance(res[61], res[291])

    # 안전 체크
    if min(
      face_width,
      face_height,
      eye_width_avg,
      interocular_distance,
      mouth_width
    ) <= 0.0001:
      logWarn("landmark 함수에서 FACE_SIZE_ERROR 발생")
      return {
        "rttype": "error",
        "errcode": "FACE_SIZE_ERROR",
        "message": "얼굴 크기가 너무 작습니다."
      }

    # =========================
    # 얼굴 비율
    # =========================
    face_ratio = face_height / face_width

    # =========================
    # 눈 관련 값
    # =========================
    eye_open_ratio = eye_height_avg / eye_width_avg
    eye_width_ratio = eye_width_avg / interocular_distance

    left_brow_y = res[296].y
    right_brow_y = res[66].y

    eyebrow_balance_ratio = (
      left_brow_y - right_brow_y
    ) / face_height

    # 눈썹 기울기
    # 50 = 중립, 50보다 작으면 내려감, 크면 올라감
    left_eyebrow_slope = res[334].y - res[296].y
    right_eyebrow_slope = -(res[105].y - res[66].y)

    eyebrow_slope_ratio = (
      (left_eyebrow_slope + right_eyebrow_slope) / 2
    ) / interocular_distance

    eyebrow_slope_score = clamp(50 + eyebrow_slope_ratio * 200)

    # 눈 비대칭
    eye_asymmetry_ratio = abs(left_eye_height - right_eye_height) / eye_width_avg

    # =========================
    # 입 관련 값
    # =========================
    mouth_open_ratio = mouth_height / mouth_width
    mouth_width_ratio = mouth_width / interocular_distance

    mouth_center_y = (res[13].y + res[14].y) / 2
    left_corner_y = res[291].y
    right_corner_y = res[61].y

    corner_avg_y = (left_corner_y + right_corner_y) / 2

    # 값이 클수록 입꼬리가 올라간 상태
    mouth_corner_lift_ratio = (
      mouth_center_y - corner_avg_y
    ) / interocular_distance

    # 입 비대칭
    mouth_asymmetry_ratio = abs(left_corner_y - right_corner_y) / interocular_distance

    # =========================
    # 코 관련 값
    # =========================
    nose_width = distance(res[98], res[327])
    nose_width_ratio = nose_width / interocular_distance

    return {
      "rttype": "success",
      "data": {
        "score_value": {
          "eye_open": round(normalize(eye_open_ratio, 0.02, 0.55), 2),
          "eye_width": round(normalize(eye_width_ratio, 0.10, 1.20), 2),
          "eye_asymmetry": round(normalize(eye_asymmetry_ratio, 0.00, 0.15), 2),

          "mouth_open": round(normalize(mouth_open_ratio, 0.00, 0.75), 2),
          "mouth_width": round(normalize(mouth_width_ratio, 0.20, 1.80), 2),
          "mouth_asymmetry": round(normalize(mouth_asymmetry_ratio, 0.00, 0.12), 2),

          "nostril_width": round(normalize(nose_width_ratio, 0.05, 0.80), 2)
        },
        "noscore_value": {
          "face_ratio": round(face_ratio, 4),
          "mouth_corner_lift": round(mouth_corner_lift_ratio, 4),
          "eyebrow_slope": round(eyebrow_slope_score, 2),
          "eyebrow_balance": round(eyebrow_balance_ratio, 4)
        }
      }
    }

  except Exception as e:
    logError(f"FaceLandmark 처리 중 오류 발생. errormsg: {str(e)}")
    return {
      "rttype": "error",
      "errcode": "LANDMARK_PROCESSING_ERROR",
      "message": f"FaceLandmark 처리 중 오류가 발생했습니다."
    }

def siglip_classify(image, texts):
  inputs = siglip_processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
  ).to(siglip_device)

  with siglip_lock:
    with torch.no_grad():
      outputs = siglip_model(**inputs)

  scores = outputs.logits_per_image[0]

  display_probs = torch.softmax(scores, dim=0)

  best_idx = int(torch.argmax(scores).item())

  return {
    "best_index": best_idx,
    "best_label": texts[best_idx],
    "best_score": float(display_probs[best_idx].item()),
    "scores": {
      texts[i]: float(display_probs[i].item())
      for i in range(len(texts))
    }
  }

def siglip(path):
  logInfo(f"siglip 함수가 호출되었습니다.")
  try:
    image = Image.open(path).convert("RGB")

    # 감정 타입 분류
    type_keys = list(FEELING_TYPES.keys())      # ["happy", "sad", ...]
    type_texts = list(FEELING_TYPES.values())   # ["a smiling face", ...]

    type_result = siglip_classify(image, type_texts)

    best_type = type_keys[type_result["best_index"]]
    best_type_text = type_texts[type_result["best_index"]]

    # 분류된 감정 타입 안에서 상세 라벨 분류
    detail_label_map = LABELS[best_type]

    detail_ko_labels = list(detail_label_map.keys())
    detail_en_labels = list(detail_label_map.values())

    detail_result = siglip_classify(image, detail_en_labels)

    best_detail_ko = detail_ko_labels[detail_result["best_index"]]
    best_detail_en = detail_en_labels[detail_result["best_index"]]

    return {
      "rttype": "success",
      "type": {
        "label": best_type,
        "analysis_label": best_type_text,
        "score": round(type_result["best_score"], 4),
        "scores": {
          type_keys[i]: round(
            type_result["scores"][type_texts[i]],
            4
          )
          for i in range(len(type_keys))
        }
      },
      "detail": {
        "label": best_detail_ko,
        "analysis_label": best_detail_en,
        "score": round(detail_result["best_score"], 4),
        "scores": {
          detail_ko_labels[i]: round(
            detail_result["scores"][detail_en_labels[i]],
            4
          )
          for i in range(len(detail_ko_labels))
        }
      }
    }

  except Exception as e:
    logWarn(f"SigLIP 처리 중 오류 발생. errormsg: {str(e)}")
    return {
      "rttype": "error",
      "errcode": "SIGLIP_PROCESSING_ERROR",
      "message": f"SigLIP 처리 중 오류가 발생했습니다."
    }

def feedback(feedbackId):
  inputdata = get_item(feedbackId)

  if inputdata is None:
    logWarn(f"{feedbackId}: 오류. 피드백 데이터를 찾을 수 없습니다. errcode: FEEDBACK_NOT_FOUND")
    return {
      "rttype": "error",
      "errcode": "FEEDBACK_NOT_FOUND",
      "message": "피드백 데이터를 찾을 수 없습니다."
    }

  mode = inputdata.get("mode")
  prompt = inputdata.get("prompt")

  if mode not in ["imgMode", "txtMode"]:
    logWarn(f"{feedbackId}: 오류. 알 수 없는 모드가 입력 되었습니다. mode: {mode}, errcode: MODE_ERROR")
    return {
      "rttype": "error",
      "errcode": "MODE_ERROR",
      "message": f"알 수 없는 모드입니다. 입력된 모드: '{str(mode)}'" 
    }

  if mode == "imgMode" and not inputdata.get("trgt"):
    logWarn(f"{feedbackId}: 오류. 목표 이미지 데이터가 존재하지 않습니다. mode: {mode}, errcode: TARGET_IMAGE_NOT_FOUND")
    return {
      "rttype": "error",
      "errcode": "TARGET_IMAGE_NOT_FOUND",
      "message": "목표 이미지 데이터가 없습니다."
    }

  if mode == "imgMode":
    progress_map = {
      "fedb_mediapipe": 15,
      "fedb_siglip": 30,
      "feedback": 85
    }
  else:
    progress_map = {
      "fedb_mediapipe": 20,
      "fedb_siglip": 50,
      "feedback": 85
    }

  set_job(
    feedbackId,
    progress=progress_map["fedb_mediapipe"],
    message="피드백 이미지의 얼굴 특징을 분석 중입니다."
  )
  logJob(f"{feedbackId}: 피드백 이미지 landmark 분석이 시작되었습니다.")
  fedb_landmark = landmark(inputdata["fedb"])
  if fedb_landmark["rttype"] != "success":
    logWarn(f"{feedbackId}: 피드백 이미지 landmark 분석 중 오류가 발생했습니다. errcode: {fedb_landmark.get('errcode')}")
    return fedb_landmark
  logJob(f"{feedbackId}: 피드백 이미지 landmark 분석 성공!")

  set_job(
    feedbackId,
    progress=progress_map["fedb_siglip"],
    message="피드백 이미지의 의미를 분석 중입니다."
  )
  logJob(f"{feedbackId}: 피드백 이미지 siglip 분석이 시작되었습니다.")
  fedb_siglip = siglip(inputdata["fedb"])
  if fedb_siglip["rttype"] != "success":
    logWarn(f"{feedbackId}: 피드백 이미지 siglip 분석 중 오류가 발생했습니다. errcode: {fedb_siglip.get('errcode')}")
    return fedb_siglip
  logJob(f"{feedbackId}: 피드백 이미지 siglip 분석 성공!")
  result = {
    "rttype": "success",
    "feedbackID": feedbackId,
    "mode": mode,
    "context": prompt,

    "mediapipe": {
      "fedb": fedb_landmark["data"]
    },

    "siglip": {
      "fedb": fedb_siglip
    },

    "feedback": None
  }

  if mode == "imgMode":
    set_job(
      feedbackId,
      progress=50,
      message="목표 이미지의 얼굴 특징을 분석 중입니다."
    )
    logJob(f"{feedbackId}: 목표 이미지 landmark 분석이 시작되었습니다.")
    trgt_landmark = landmark(inputdata["trgt"])
    if trgt_landmark["rttype"] != "success":
      logWarn(f"{feedbackId}: 목표 이미지 landmark 분석 중 오류가 발생했습니다. errcode: {trgt_landmark.get('errcode')}")
      return trgt_landmark
    logJob(f"{feedbackId}: 목표 이미지 landmark 분석 성공!")

    set_job(
      feedbackId,
      progress=70,
      message="목표 이미지의 의미를 분석 중입니다."
    )
    logJob(f"{feedbackId}: 목표 이미지 siglip 분석이 시작되었습니다.")
    trgt_siglip = siglip(inputdata["trgt"])
    if trgt_siglip["rttype"] != "success":
      logWarn(f"{feedbackId}: 목표 이미지 siglip 분석 중 오류가 발생했습니다. errcode: {trgt_siglip.get('errcode')}")
      return trgt_siglip
    logJob(f"{feedbackId}: 목표 이미지 siglip 분석 성공!")

    result["mediapipe"]["trgt"] = trgt_landmark["data"]
    result["siglip"]["trgt"] = trgt_siglip

  else:
    result["mediapipe"]["trgt"] = None
    result["siglip"]["trgt"] = None

  if LLM_SWITCH == "On":
    set_job(
      feedbackId,
      progress=85,
      message="최종 피드백을 생성 중입니다."
    )

    if(mode == "imgMode"):
      try:
        llm_user_msg = f"""mode: {mode},
          User`s prompt: {prompt},

          == features of feedback images ==
          face_ratio: {result['mediapipe']['fedb']['noscore_value']['face_ratio']}
          eye_open: {result['mediapipe']['fedb']['score_value']['eye_open']}
          eye_width: {result['mediapipe']['fedb']['score_value']['eye_width']}
          eye_asymmetry: {result['mediapipe']['fedb']['score_value']['eye_asymmetry']}

          mouth_open: {result['mediapipe']['fedb']['score_value']['mouth_open']}
          mouth_width: {result['mediapipe']['fedb']['score_value']['mouth_width']}
          mouth_corner_lift: {result['mediapipe']['fedb']['noscore_value']['mouth_corner_lift']}
          mouth_asymmetry: {result['mediapipe']['fedb']['score_value']['mouth_asymmetry']}

          nostril_width: {result['mediapipe']['fedb']['score_value']['nostril_width']}

          eyebrow_slope: {result['mediapipe']['fedb']['noscore_value']['eyebrow_slope']}
          eyebrow_balance: {result['mediapipe']['fedb']['noscore_value']['eyebrow_balance']}

          == speculated emotional information of the feedback image ==
          guessed feeling: {result['siglip']['fedb']['type']['label']}
          feeling analysis label: {result['siglip']['fedb']['type']['analysis_label']}
          feeling accuracy: {result['siglip']['fedb']['type']['score'] * 100:.2f}%

          guessed facial detail: {result['siglip']['fedb']['detail']['label']}
          detail analysis label: {result['siglip']['fedb']['detail']['analysis_label']}
          detail accuracy: {result['siglip']['fedb']['detail']['score'] * 100:.2f}%

          emotion distribution:
          {result['siglip']['fedb']['type']['scores']}

          detail distribution:
          {result['siglip']['fedb']['detail']['scores']}
        """
      except Exception as e:
        logError(f"{feedbackId}, imgMode: LLM 프롬프트 완성 중 에러 발생. errcode: {str(e)}")
        return {
          "rttype": "error",
          "errcode": "LLM_PROMPT_ERROR",
          "message": f"프롬프트 완성 중 오류가 발생했습니다."
        }
    elif(mode == "txtMode"):
      try:
        llm_user_msg = f"""mode: {mode},
          User`s prompt: {prompt},

          == features of feedback images ==
          face_ratio: {result['mediapipe']['fedb']['noscore_value']['face_ratio']}
          eye_open: {result['mediapipe']['fedb']['score_value']['eye_open']}
          eye_width: {result['mediapipe']['fedb']['score_value']['eye_width']}
          eye_asymmetry: {result['mediapipe']['fedb']['score_value']['eye_asymmetry']}

          mouth_open: {result['mediapipe']['fedb']['score_value']['mouth_open']}
          mouth_width: {result['mediapipe']['fedb']['score_value']['mouth_width']}
          mouth_corner_lift: {result['mediapipe']['fedb']['noscore_value']['mouth_corner_lift']}
          mouth_asymmetry: {result['mediapipe']['fedb']['score_value']['mouth_asymmetry']}

          nostril_width: {result['mediapipe']['fedb']['score_value']['nostril_width']}

          eyebrow_slope: {result['mediapipe']['fedb']['noscore_value']['eyebrow_slope']}
          eyebrow_balance: {result['mediapipe']['fedb']['noscore_value']['eyebrow_balance']}

          == speculated emotional information of the feedback image ==
          guessed feeling: {result['siglip']['fedb']['type']['label']}
          feeling analysis label: {result['siglip']['fedb']['type']['analysis_label']}
          feeling accuracy: {result['siglip']['fedb']['type']['score'] * 100:.2f}%

          guessed facial detail: {result['siglip']['fedb']['detail']['label']}
          detail analysis label: {result['siglip']['fedb']['detail']['analysis_label']}
          detail accuracy: {result['siglip']['fedb']['detail']['score'] * 100:.2f}%

          emotion distribution:
          {result['siglip']['fedb']['type']['scores']}

          detail distribution:
          {result['siglip']['fedb']['detail']['scores']}
        """
      except Exception as e:
        logError(f"{feedbackId}, txtMode: LLM 프롬프트 완성 중 에러 발생. errcode: {str(e)}")
        return {
          "rttype": "error",
          "errcode": "LLM_PROMPT_ERROR",
          "message": f"프롬프트 완성 중 오류가 발생했습니다."
        }
    logJob(f"{feedbackId}: {LLM_MODEL}를 사용하여 피드백 생성 중...")
    if(LLM_MODEL == "gemini"):
      try:
        response = gemini_client.models.generate_content(
          model=GEMINI_MODELNAME,
          config=types.GenerateContentConfig(
            system_instruction=LLM_PROMPT,
          ),
          contents=llm_user_msg
        ).text
      except Exception as e:
        logWarn(f"{feedbackId}: 피드백 생성 중 오류 발생. errcode: LLM_PROCESSING_ERROR, errormsg: {str(e)}")
        return {
          "rttype": "error",
          "errcode": "LLM_PROCESSING_ERROR",
          "message": f"피드백 생성 중 오류가 발생했습니다"
        }
    elif(LLM_MODEL == "gpt"):
      try:
        response = gpt_client.responses.create(
          model=GPT_MODELNAME,
          input=[
            {
                "role": "system",
                "content": LLM_PROMPT
            },
            {
                "role": "user",
                "content": llm_user_msg
            }
          ]
        ).output_text
      except Exception as e:
        logWarn(f"{feedbackId}: 피드백 생성 중 오류 발생. errcode: LLM_PROCESSING_ERROR, errormsg: {str(e)}")
        return {
          "rttype": "error",
          "errcode": "LLM_PROCESSING_ERROR",
          "message": f"피드백 생성 중 오류가 발생했습니다"
        }
    logJob(f"{feedbackId}: 피드백 생성 성공!")
    result["feedback"] = response
  else:
    result["feedback"] = "서비스 운영자가 LLM 피드백을 비활성화 시켰습니다."

  return result


def set_job(feedbackId, **kwargs):
  with jobs_lock:
    if feedbackId not in jobs:
      return

    jobs[feedbackId].update(kwargs)

def worker():
  global processing_count

  while True:
    feedbackId = job_queue.get()

    with processing_lock:
      processing_count += 1

    try:
      logJob(f"{feedbackId}: 작업이 시작되었습니다. Status: processing")
      set_job(
        feedbackId,
        status="processing",
        progress=10,
        message="분석을 시작했습니다."
      )

      logJob(f"{feedbackId}: feedback 함수를 실행하는 중입니다.")
      result = feedback(feedbackId)

      if result.get("rttype") == "success":
        logJob(f"{feedbackId}: 정상적으로 분석이 완료되었습니다.")
        set_job(
          feedbackId,
          status="done",
          progress=100,
          message="분석이 완료되었습니다.",
          result=result
        )
      else:
        logJob(f"{feedbackId}: 분석 중 오류가 발생했습니다.")
        set_job(
          feedbackId,
          status="error",
          progress=100,
          message="분석 중 오류가 발생했습니다.",
          error=result
        )

    except Exception as e:
      logWarn(f"500, Worker Error: {str(e)}")
      set_job(
        feedbackId,
        status="error",
        progress=100,
        message="서버 내부 오류가 발생했습니다.",
        error={
          "rttype": "error",
          "errcode": "INTERNAL_SERVER_ERROR"
        }
      )

    finally:
      with processing_lock:
        processing_count -= 1

      job_queue.task_done()

app = Flask(__name__)

from flask import request

@app.before_request
def log_all_requests():
  if COSTOM_LOGGING == "On":
    real_ip = (
      request.headers.get("CF-Connecting-IP")
      or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
      or request.remote_addr
    )

    print(
      f'{Fore.LIGHTGREEN_EX}[USER]{Style.RESET_ALL} [IP: {real_ip}] '
      f'{request.path} 요청. route: "{request.path}" | Method: {request.method}'
    )


@app.errorhandler(404)
def page_not_found(e):
  real_ip = (
    request.headers.get("CF-Connecting-IP")
    or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    or request.remote_addr
  )
  print(
    f'[USER] [IP: {real_ip}] '
    f'404 요청. route: "{request.path}" | Method: {request.method}',
    flush=True
  )

  return render_template("404.html"), 404

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/i")
def i():
  return render_template("install.html")

@app.route('/sw.js')
def sw():
  return send_from_directory(
                              'static',
                              'js/sw.js', 
                              mimetype='application/javascript'
                            )

@app.route("/api/feedback", methods=["POST"])
def api_feedback():
  real_ip = (
    request.headers.get("CF-Connecting-IP")
    or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    or request.remote_addr
  )
  time_data = request.form.get('time')
  if(time_data is None):
    logUser(f"[IP: {real_ip}] \"/api/feedback\": 'time' 데이터 누락. [400, Bad Request] 반환 처리 됨.")
    return {"code": 400, "message": "Bad request."}, 400
  feedbackId = str(uuid.uuid4())

  mode = request.form.get("mode")

  feedback_dir = os.path.join(UPLOAD_DIR, feedbackId)
  os.makedirs(feedback_dir, exist_ok=True)

  if(mode == "imgMode"):
    fedb_img = request.files.get('fedbImg')
    trgt_img = request.files.get('trgtImg')
    prompt = request.form.get('prompt')

    if prompt is None:
      prompt = "[사용자가 입력하지 않음]"
    
    if fedb_img is None or trgt_img is None:
      return {"code": 400, "message": "missing image file"}, 400
    
    fedb_ext = os.path.splitext(fedb_img.filename)[1].lower()
    fedb_path = os.path.join(feedback_dir, f"{feedbackId}_f{fedb_ext}")
    
    trgt_ext = os.path.splitext(trgt_img.filename)[1].lower()
    trgt_path = os.path.join(feedback_dir, f"{feedbackId}_t{trgt_ext}")

    if fedb_ext not in ALLOWED_EXTENSIONS or trgt_ext not in ALLOWED_EXTENSIONS:
      logInfo(f"[보안 경고] \"/api/feedback\": 허용하지 않는 파일 확장자 입력됨. 확장자: <fedb: {fedb_ext}>, <trgt: {trgt_ext}> [400, disallowed file type] 반환 처리 됨.")
      return {"code": 400, "message": "disallowed file type"}, 400

    if fedb_img:
      fedb_img.save(fedb_path)

    if trgt_img:
      trgt_img.save(trgt_path)

    add_item(feedbackId, {
      "mode": mode,
      "fedb": fedb_path,
      "trgt": trgt_path,
      "prompt": prompt,
      "time": time_data,
      "ipAddr": real_ip
    })
  elif(mode == "txtMode"):
    fedb_img = request.files.get("fedbImg")
    prompt = request.form.get('prompt')
    time_data = request.form.get('time')

    if prompt is None:
      return {"code": 400, "message": "missing prompt text"}, 400

    if fedb_img is None:
      return {"code": 400, "message": "missing image file"}, 400

    fedb_ext = os.path.splitext(fedb_img.filename)[1].lower()
    fedb_path = os.path.join(feedback_dir, f"{feedbackId}_f{fedb_ext}")

    if fedb_ext not in ALLOWED_EXTENSIONS:
      logWarn(f"[보안 경고] \"/api/feedback\": 허용하지 않는 파일 확장자 입력 됨. 확장자: <fedb: {fedb_ext}> [400, disallowed file type] 반환 처리 됨.")
      return {"code": 400, "message": "disallowed file type"}, 400

    if fedb_img:
      fedb_img.save(fedb_path)

    add_item(feedbackId, {
      "mode": mode,
      "fedb": fedb_path,
      "prompt": prompt,
      "time": time_data,
      "ipAddr": real_ip
    })
  else:
    logInfo(f"\"/api/feedback\": 'mode' 데이터 누락. [400, mode error] 반환 처리 됨.")
    return {"code": 400, "message": "mode error"}, 400

  logJob(f"{feedbackId}: 새로운 작업이 대기열에 추가되었습니다.")
  job_queue.put(feedbackId)

  queue_list = list(job_queue.queue)
  queue_ahead = queue_list.index(feedbackId)
  with jobs_lock:
    jobs[feedbackId] = {
      "feedbackId": feedbackId,
      "status": "queued",
      "queue_ahead": queue_ahead,
      "progress": 0,
      "message": "대기열에 등록되었습니다.",
      "result": None,
      "error": None,
      "created_at": time.time()
    }

  return jsonify({
    "rttype": "success",
    "feedbackId": feedbackId,
    "status": "queued",
    "message": "대기열에 등록되었습니다."
  })

@app.route("/api/feedback/stream/<feedbackId>", methods=["GET"])
def api_feedback_stream(feedbackId):
  def event_stream():
    while True:
      with jobs_lock:
        job = jobs.get(feedbackId)

      if job is None:
        payload = {
          "rttype": "error",
          "errcode": "JOB_NOT_FOUND",
          "message": "작업을 찾을 수 없습니다."
        }
      else:
        with job_queue.mutex:
          queue_list = list(job_queue.queue)

          if job["status"] == "queued":
            try:
              idx = queue_list.index(feedbackId)
              queue_ahead = idx
            except ValueError:
              queue_ahead = -1
          else:
            queue_ahead = 0

          payload = dict(job)
          payload["queue_ahead"] = queue_ahead

      yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

      if job is None:
        logJob(f"{feedbackId}: 작업을 찾을 수 없어 SSE 종료 됨.")
        break

      if job["status"] in ["done", "error"]:
        if job["status"] == "done":
          logJob(f"{feedbackId}: 작업이 완료되어 SSE 종료 됨.")
        elif job["status"] == "error":
          logJob(f"{feedbackId}: 작업 처리 중 오류가 발생하여 SSE 종료 됨.")
        break

      time.sleep(0.5)

  return Response(event_stream(), mimetype="text/event-stream")

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

def printStartMSG():
  print()
  print()
  print(f"{Fore.LIGHTBLUE_EX}서비스가 시작되었습니다!{Style.RESET_ALL}")
  print()
  print(f"프로젝트 디렉토리: {Fore.GREEN}{PROJECT_DIR}{Style.RESET_ALL}")
  print(f"커스텀 로그: {Fore.GREEN}{COSTOM_LOGGING}{Style.RESET_ALL}")
  print()
  print(f"서비스 URL: {Fore.GREEN}http://127.0.0.1:{SERVICE_PORT}{Style.RESET_ALL}\n            {Fore.GREEN}http://{get_local_ip()}:{SERVICE_PORT}{Style.RESET_ALL}")
  print()
  if LLM_MODEL == "gpt":
    print(f"{Fore.RED}{Style.BRIGHT}※주의※ {Fore.YELLOW}현재 GPT로 서비스가 실행되고 있습니다.{Style.RESET_ALL}")
    print()


server_ready = False
def wait_until_server_ready():
  global server_ready
  import socket, time
  while True:
    try:
      with socket.create_connection(("127.0.0.1", SERVICE_PORT), timeout=1):
        server_ready = True
        printStartMSG()
        break
    except:
      time.sleep(0.1)

if __name__ == '__main__':
  Thread(target=worker, daemon=True).start()
  Thread(target=wait_until_server_ready, daemon=True).start()
  app.run(host="0.0.0.0", port=SERVICE_PORT, use_reloader=False)