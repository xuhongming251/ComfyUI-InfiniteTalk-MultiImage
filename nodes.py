

import torchaudio
import torch
import numpy as np
from tqdm import tqdm
import webrtcvad
from comfy.utils import common_upscale
import datetime


class GetFloatByIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_batch": ("FLOAT", ),  # 接受 FLOAT batch
                "index": ("INT", { "default": 0, "min": 0 }),
            }
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("value", )
    FUNCTION = "get_value"
    CATEGORY = "InfiniteTalk"

    def get_value(self, float_batch, index):
        # float_batch 可能是 tensor 或 Python list
        if hasattr(float_batch, "tolist"):
            data = float_batch.tolist()
        else:
            data = list(float_batch)

        if len(data) == 0:
            return (0.0,)  # 默认返回 float

        safe_index = max(0, min(index, len(data) - 1))

        return (float(data[safe_index]),)

class MakeBatchFromFloatList:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_list": ("FLOAT",),
            }
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("float_batch", )
    INPUT_IS_LIST = True
    FUNCTION = "make_batch"
    CATEGORY = "InfiniteTalk"

    def make_batch(self, float_list):

        # 一个或没有 → 直接返回
        if len(float_list) <= 1:
            return (float_list,)

        # 转 tensor batch
        float_tensor = torch.tensor(float_list, dtype=torch.float32)

        return (float_tensor,)


class GetIntByIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "int_batch": ("INT", ),   # 接受 INT batch
                "index": ("INT", { "default": 0, "min": 0 }),
            }
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("value", )
    FUNCTION = "get_value"
    CATEGORY = "InfiniteTalk"

    def get_value(self, int_batch, index):
        # int_batch 可能是 tensor，也可能是 Python list
        # 统一处理为 Python list
        if hasattr(int_batch, "tolist"):
            data = int_batch.tolist()
        else:
            data = list(int_batch)

        # 越界保护
        if len(data) == 0:
            return (0,)  # 无数据默认返回 0

        safe_index = max(0, min(index, len(data) - 1))

        return (int(data[safe_index]),)

class MakeBatchFromIntList:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "int_list": ("INT",),
            }
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("int_batch", )
    INPUT_IS_LIST = True
    FUNCTION = "make_batch"
    CATEGORY = "InfiniteTalk"

    def make_batch(self, int_list):

        # 只有一个或没有 → 直接返回
        if len(int_list) <= 1:
            return (int_list,)

        # 转为 tensor batch
        # int_list 是 Python int 列表，需要转成 tensor
        int_tensor = torch.tensor(int_list, dtype=torch.int32)

        return (int_tensor,)
    

from decimal import Decimal, getcontext, ROUND_HALF_UP

# 设置高精度（足够即可）
getcontext().prec = 28

def align_step_004(x: Decimal) -> Decimal:
    """保证 x 是 0.04 的倍数，不丢精度"""
    step = Decimal("0.04")
    q = (x / step).to_integral_value(rounding=ROUND_HALF_UP)
    return q * step


class InfiniteTalkMultiImage():

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "audio_duration": ("FLOAT", {"default": 8.0, "min": 0.01, "step": 0.04}),
                "prompt_list_input": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "auto_start_time_list": ("START_TIME_LIST", {"default": [], "forceInput": True}),
            }
        }

        # 默认 start_time: 0, 3, 6, ... 但内部仍用 Decimal 处理
        for i in range(1, 21):
            inputs["optional"][f"image{i}"] = ("IMAGE",)
            default_start = (i - 1) * 3.0
            inputs["required"][f"start_time{i}"] = (
                "FLOAT",
                {"default": default_start, "min": 0.0, "step": 0.04}
            )

        return inputs

    OUTPUT_IS_LIST = (True, True, True, True)
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("image_list", "frame_count_list", "real_start_time_list", "prompt_list")
    FUNCTION = "calculate_big_loops"
    CATEGORY = "InfiniteTalk"

    @staticmethod
    def calculate_big_loops(**kwargs):
        fps = Decimal("25")
        
        auto_start_time_list = kwargs.get("auto_start_time_list", [])

        print("auto_start_time_list:", auto_start_time_list)

        # =============== 音频长度 Decimal 无损读取 ===============
        audio_duration = Decimal(str(kwargs.get("audio_duration", "0")))
        audio_duration = align_step_004(audio_duration)

        # =============== 文本 ===============
        raw_text = kwargs.get("prompt_list_input", "")
        text_list = [line.strip() for line in raw_text.split("\n") if line.strip()]

        # =============== 图片 + start_time 数据收集 ===============
        pairs = []

        for i in range(1, 21):
            img = kwargs.get(f"image{i}", None)
            raw_st = kwargs.get(f"start_time{i}", None)

            if img is not None:
                st = Decimal(str(raw_st))
                st = align_step_004(st)  # 强制对齐到 0.04
                pairs.append((st, img))

        if not pairs:
            return [], [], [], text_list

        # 分离出初步的时间和图片列表
        raw_start_times = [p[0] for p in pairs]
        raw_images = [p[1] for p in pairs]

        # 如果有自动时间列表，覆盖手动输入的时间
        if auto_start_time_list is not None and len(auto_start_time_list) > 0:
            raw_start_times = [Decimal(str(t)) for t in auto_start_time_list]

        # 确保图片和时间列表长度一致（取最小值）
        min_len = min(len(raw_images), len(raw_start_times))
        raw_images = raw_images[:min_len]
        raw_start_times = raw_start_times[:min_len]

        # =============== 核心修改：根据音频时长过滤/对齐 ===============
        images = []
        start_times = []
        
        # 遍历所有待选片段，只保留开始时间早于音频总时长的片段
        for img, st in zip(raw_images, raw_start_times):
            if st < audio_duration:
                images.append(img)
                start_times.append(st)
            else:
                # 一旦发现起始时间超过或等于音频时长，后面的通常也都不需要了
                # 这里不做 break 是为了防止输入时间乱序的情况，虽然通常是顺序的
                continue
        
        # 如果过滤后为空（例如音频极短，第一张图开始时间都比音频长）
        if not start_times:
            # 返回空列表或默认处理，防止后续报错
            # 这里选择返回空，或者您可以选择至少保留第一张图给 0 帧
            return [], [], [], [] 

        # =============== frame count（真实差值） ===============
        frame_count_list = []

        for idx, st in enumerate(start_times):
            # 确定当前片段的结束时间
            if idx < len(start_times) - 1:
                # 如果不是最后一张，结束时间通常是下一张的开始时间
                end = start_times[idx + 1]
                
                # 【保护措施】如果下一张图的开始时间意外超过了音频时长（虽然前面过滤过，但防止乱序）
                # 或者逻辑上我们希望片段不超出音频范围
                if end > audio_duration:
                    end = audio_duration
            else:
                # 如果是最后一张，结束时间就是音频总时长
                end = audio_duration

            dur = end - st
            if dur < 0:
                dur = Decimal("0")

            frames = int(dur * fps)  # Decimal → int 无损
            frame_count_list.append(frames)

        # =============== real_start_time_list ===============
        real_start_time_list = [float(t) for t in start_times]

        # =============== prompt 对齐 ===============
        # 此时 images 已经是被 audio_duration 过滤后的列表
        n_images = len(images)
        n_texts = len(text_list)

        if n_texts < n_images:
            # 如果文本数量不够，用空字符串填充
            text_list = text_list + [""] * (n_images - n_texts)
        elif n_texts > n_images:
            # 如果文本数量多于（过滤后的）图片数量，截断多余部分
            text_list = text_list[:n_images]

        return images, frame_count_list, real_start_time_list, text_list


class InfiniteTalkEmbedsSlice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multitalk_embeds": ("MULTITALK_EMBEDS",),
                "start_video_frame": ("INT", {"default": 0, "min": 0}),
                "video_frame_length": ("INT", {"default": 30, "min": 1}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 120}),
            }
        }

    RETURN_TYPES = ("MULTITALK_EMBEDS",)
    RETURN_NAMES = ("sliced_multitalk_embeds",)
    FUNCTION = "slice"
    CATEGORY = "WanVideoWrapper"

    def slice(self, multitalk_embeds, start_video_frame, video_frame_length, fps):
        """
        将视频帧区间转换为音频 embedding 帧区间：
        腾讯 wav2vec2 每秒 50 帧 → audio_fps = 50
        所以：
            audio_frame = int(video_frame * audio_fps / fps)
                         = int(video_frame * (50 / fps))
        fps 默认=25 → 每个视频帧 = 2 音频帧
        """

        # MultiTalk wav2vec2 输出固定 50 帧/秒
        AUDIO_FPS = 25
        

        # === 计算音频帧区间 ===
        start_audio_frame = int(start_video_frame * AUDIO_FPS / fps)
        audio_frame_length = int(video_frame_length * AUDIO_FPS / fps)

        audio_features = multitalk_embeds["audio_features"]
        audio_scale = multitalk_embeds.get("audio_scale", 1.0)
        audio_cfg_scale = multitalk_embeds.get("audio_cfg_scale", 1.0)
        ref_target_masks = multitalk_embeds.get("ref_target_masks", None)

        sliced_features = []

        for emb in audio_features:
            total_audio_frames = emb.shape[0]

            if start_audio_frame >= total_audio_frames:
                sliced = emb.new_zeros((0, emb.shape[1], emb.shape[2]))
            else:
                end_audio_frame = min(start_audio_frame + audio_frame_length, total_audio_frames)
                sliced = emb[start_audio_frame:end_audio_frame]

            sliced_features.append(sliced)

        sliced_embeds = {
            "audio_features": sliced_features,
            "audio_scale": audio_scale,
            "audio_cfg_scale": audio_cfg_scale,
            "ref_target_masks": ref_target_masks,
        }

        return (sliced_embeds,)




# # --- 加载 Silero VAD ---
# vad_model, utils = torch.hub.load(
#     repo_or_dir='snakers4/silero-vad',
#     model='silero_vad',
#     force_reload=False,
#     onnx=False,
# )

# (get_speech_timestamps,
#  save_audio,
#  read_audio,
#  VADIterator,
#  collect_chunks) = utils


# # ================================================================
# #                      ComfyUI 节点定义
# # ================================================================
# class AudioVADNode:
#     """
#     使用 Silero VAD 检测音频中的人声段落
#     输出：
#         - segments：数组，每项为 {"start": 秒, "end": 秒, "duration": 秒}
#         - label_txt：一个 txt 文件路径，包含所有人声起止点
#     """

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "audio_path": ("STRING", {"default": "", "multiline": False}),
#             }
#         }

#     RETURN_TYPES = ("DICT", "STRING")
#     RETURN_NAMES = ("segments", "label_file")
#     FUNCTION = "run"
#     CATEGORY = "Audio/Analysis"

#     def run(self, audio_path):
#         if not os.path.exists(audio_path):
#             raise Exception(f"Audio file not found: {audio_path}")

#         # --- 读取音频 ---
#         wav = read_audio(audio_path)
#         sr = 16000

#         # --- VAD 检测 ---
#         timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sr)

#         segments = []
#         for ts in timestamps:
#             start = ts['start'] / sr
#             end = ts['end'] / sr
#             duration = end - start
#             segments.append({
#                 "start": round(start, 3),
#                 "end": round(end, 3),
#                 "duration": round(duration, 3)
#             })

#         # --- 输出 label 文件 ---
#         output_dir = Path(os.getcwd()) / "vad_output"
#         output_dir.mkdir(exist_ok=True)

#         label_file = output_dir / f"vad_segments.txt"
#         with open(label_file, "w", encoding="utf-8") as f:
#             for i, seg in enumerate(segments):
#                 f.write(f"{seg['start']}\t{seg['end']}\tsegment_{i+1}\n")

#         return segments, str(label_file)



class AudioSmartSlice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "min_sec": ("FLOAT", {"default": 10.0, "min": 1.0}),
                "max_sec": ("FLOAT", {"default": 12.0, "min": 1.0}),
                "vad_aggressiveness": ("INT", {"default": 2, "min": 0, "max": 3}),
                "frame_ms": ("INT", {"default": 30, "min": 10, "max": 50}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("chunk_start_times", "chunk_durations")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "run"
    CATEGORY = "audio/slicing"

    def run(self, audio, min_sec, max_sec, vad_aggressiveness, frame_ms):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # ----------------------------
        # 1) 兼容官方 AUDIO shape
        # ----------------------------
        wf = waveform
        if wf.ndim == 3:  # (1, C, T)
            wf = wf.mean(dim=1)
        if wf.ndim == 2:  # (1, T)
            wf = wf.squeeze(0)

        # 必须重采样到 16000Hz 才能用于 webrtcvad
        if sample_rate != 16000:
            wf = torchaudio.functional.resample(wf, orig_freq=sample_rate, new_freq=16000)
            sample_rate = 16000

        wf_np = wf.detach().cpu().numpy()
        total_samples = len(wf_np)

        # ----------------------------
        # 2) webrtcvad 需要 16-bit PCM
        # ----------------------------
        wf_int16 = np.int16(np.clip(wf_np * 32768, -32768, 32767))

        # ----------------------------
        # 3) 分帧
        # ----------------------------
        frame_bytes = int(sample_rate * frame_ms / 1000)
        frames = []
        for start in range(0, total_samples, frame_bytes):
            end = min(start + frame_bytes, total_samples)
            chunk = wf_int16[start:end]
            # 补齐最后一帧
            if len(chunk) < frame_bytes:
                chunk = np.pad(chunk, (0, frame_bytes - len(chunk)), mode='constant')
            frames.append(chunk.tobytes())

        # ----------------------------
        # 4) VAD 检测人声
        # ----------------------------
        vad = webrtcvad.Vad(vad_aggressiveness)
        voiced_flags = [vad.is_speech(f, sample_rate) for f in frames]

        # ----------------------------
        # 5) 找静音段 (sample indices)
        # ----------------------------
        silent_segments = []
        start_frame = None
        for i, flag in enumerate(voiced_flags):
            if not flag and start_frame is None:
                start_frame = i
            elif flag and start_frame is not None:
                silent_segments.append((start_frame, i))
                start_frame = None
        if start_frame is not None:
            silent_segments.append((start_frame, len(voiced_flags)))

        # 转换成样本位置 (start_sample, end_sample)
        silent = [(s[0] * frame_bytes, s[1] * frame_bytes) for s in silent_segments]

        # ----------------------------
        # 6) 切片逻辑 (修复版)
        # ----------------------------
        target_min = int(min_sec * sample_rate)
        target_max = int(max_sec * sample_rate)

        chunks_start_sec = []
        chunks_dur_sec = []

        pos = 0
        pbar = tqdm(total=total_samples, desc="Processing chunks", unit="samples")
        
        while pos < total_samples:
            # 这里的逻辑是：必须切在 [pos + min, pos + max] 之间
            min_cut_point = pos + target_min
            max_cut_point = pos + target_max
            
            # 如果最小切点已经超过总长度，直接把剩下的包圆了
            if min_cut_point >= total_samples:
                chunks_start_sec.append(pos / sample_rate)
                chunks_dur_sec.append((total_samples - pos) / sample_rate)
                pbar.update(total_samples - pos)
                break

            # 默认强制在 max 处切断 (Hard Cut)
            actual_cut_point = min(max_cut_point, total_samples)

            # 尝试在 [min_cut_point, max_cut_point] 范围内寻找静音点
            found_silence = False
            
            for s_start, s_end in silent:
                # 优化：如果静音段完全在当前窗口之前，跳过
                if s_end < min_cut_point:
                    continue
                # 优化：如果静音段开始得太晚（超过了最大限制），后面的都不用看了
                if s_start > actual_cut_point:
                    break

                # 情况 A: 刚好有一段静音涵盖了 min_cut_point
                # 此时我们可以在 min_cut_point 处安全切断（因为那里是静音）
                if s_start <= min_cut_point <= s_end:
                    actual_cut_point = min_cut_point
                    found_silence = True
                    break

                # 情况 B: 有一段静音的开头落在了 [min, max] 区间内
                if min_cut_point <= s_start <= actual_cut_point:
                    actual_cut_point = s_start
                    found_silence = True
                    break 

            # 记录分段
            duration = actual_cut_point - pos
            chunks_start_sec.append(pos / sample_rate)
            chunks_dur_sec.append(duration / sample_rate)
            
            pbar.update(duration)
            
            # 更新 pos
            # 这里保证了 actual_cut_point >= pos + target_min (除非到了文件末尾)
            # 所以绝对不会死循环
            pos = actual_cut_point

        pbar.close()

        return (chunks_start_sec, chunks_dur_sec)
    


import os
import glob
import subprocess
import shutil
import tempfile
import torch
import torchaudio
from PIL import Image
import numpy as np

import folder_paths
# ========== 全部必要 imports ==========
import os
import glob
import shutil
import subprocess
import tempfile
import random
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageEnhance

import torch
import torchaudio

import folder_paths

# ========== VideoFromPathsAndAudio 节点（完整） ==========
class VideoFromPathsAndAudio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_paths": ("STRING", {"multiline": True, "default": "", "placeholder": "每行一个路径 (文件夹或文件)"}),
                "frame_counts": ("STRING", {"multiline": True, "default": "", "placeholder": "每行对应路径的帧数"}),
                "audio": ("AUDIO",),
                "fps": ("INT", {"default": 25, "min": 1, "max": 120, "step": 1}),
                "filename_prefix": ("STRING", {"default": "video_output"}),
                "format": (["mp4", "mkv", "mov"],),

                # 新增过渡参数（严格保持你要求的名字和位置）
                "enable_transition": ("BOOLEAN", {"default": True}),
                "transition_seconds": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 5.0, "step": 0.01}),
                "transition_type": (
                    [
                        "crossfade",
                        "left_wipe",
                        "right_wipe",
                        # "up_wipe",
                        # "down_wipe",
                        "slide_left",
                        "slide_right",
                        # "slide_up",
                        # "slide_down",
                        "zoom_in",
                        "zoom_out",
                        "soft_blur",
                        "random",
                    ],
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "execute_video_synth"
    CATEGORY = "Custom/Video"

    # ----------------------------
    # 打开两张图片并保持大小一致
    # ----------------------------
    def _open_and_match(self, pathA, pathB):
        A = Image.open(pathA).convert("RGB")
        B = Image.open(pathB).convert("RGB")
        if A.size != B.size:
            B = B.resize(A.size, Image.LANCZOS)
        return A, B

    # ----------------------------
    # 缓动函数（Easing Functions）- 让过渡更自然
    # ----------------------------
    def _ease_in_out_cubic(self, t):
        """平滑的缓入缓出曲线"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    def _ease_out_cubic(self, t):
        """缓出曲线"""
        return 1 - pow(1 - t, 3)
    
    def _ease_in_out_quad(self, t):
        """二次缓入缓出"""
        if t < 0.5:
            return 2 * t * t
        else:
            return 1 - pow(-2 * t + 2, 2) / 2
    
    def _smooth_step(self, t):
        """平滑步进函数"""
        return t * t * (3 - 2 * t)

    # ----------------------------
    # 过渡：crossfade（优化版 - 使用缓动和更好的混合）
    # ----------------------------
    def _crossfade(self, A, B, t):
        # 使用缓动函数让过渡更平滑
        t_smooth = self._ease_in_out_cubic(t)
        # 使用更自然的混合
        return Image.blend(A, B, t_smooth)

    # ----------------------------
    # 过渡：left/right/up/down wipe（优化版 - 添加渐变边缘）
    # ----------------------------
    def _wipe(self, A, B, t, direction):
        w, h = A.size
        # 使用缓动函数
        t_smooth = self._ease_in_out_quad(t)
        
        # 创建渐变边缘的 mask（让擦除更柔和）
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        gradient_width = max(20, min(w, h) // 10)  # 渐变宽度为画面的10%，最少20像素
        
        if direction == "left":
            cut = int(w * t_smooth)
            # 填充已擦除区域（完全显示B的部分）
            if cut > gradient_width:
                draw.rectangle([0, 0, cut - gradient_width, h], fill=255)
            # 创建渐变区域
            for x in range(max(0, cut - gradient_width), min(w, cut + gradient_width)):
                if x < cut:
                    alpha = 255
                else:
                    # 渐变边缘
                    dist = x - cut
                    alpha = max(0, 255 - int(255 * dist / gradient_width))
                draw.rectangle([x, 0, x + 1, h], fill=alpha)
        elif direction == "right":
            cut = int(w * t_smooth)
            # 填充已擦除区域
            if cut > gradient_width:
                draw.rectangle([w - cut + gradient_width, 0, w, h], fill=255)
            # 创建渐变区域
            for x in range(max(0, w - cut - gradient_width), min(w, w - cut + gradient_width)):
                if x >= w - cut:
                    alpha = 255
                else:
                    dist = (w - cut) - x
                    alpha = max(0, 255 - int(255 * dist / gradient_width))
                draw.rectangle([x, 0, x + 1, h], fill=alpha)
        elif direction == "up":
            cut = int(h * t_smooth)
            # 填充已擦除区域
            if cut > gradient_width:
                draw.rectangle([0, 0, w, cut - gradient_width], fill=255)
            # 创建渐变区域
            for y in range(max(0, cut - gradient_width), min(h, cut + gradient_width)):
                if y < cut:
                    alpha = 255
                else:
                    dist = y - cut
                    alpha = max(0, 255 - int(255 * dist / gradient_width))
                draw.rectangle([0, y, w, y + 1], fill=alpha)
        else:  # down
            cut = int(h * t_smooth)
            # 填充已擦除区域
            if cut > gradient_width:
                draw.rectangle([0, h - cut + gradient_width, w, h], fill=255)
            # 创建渐变区域
            for y in range(max(0, h - cut - gradient_width), min(h, h - cut + gradient_width)):
                if y >= h - cut:
                    alpha = 255
                else:
                    dist = (h - cut) - y
                    alpha = max(0, 255 - int(255 * dist / gradient_width))
                draw.rectangle([0, y, w, y + 1], fill=alpha)
        
        return Image.composite(B, A, mask)

    # ----------------------------
    # 过渡：slide（优化版 - 使用缓动和边缘混合）
    # ----------------------------
    def _slide(self, A, B, t, direction):
        w, h = A.size
        # 使用缓动函数让滑动更平滑
        t_smooth = self._ease_in_out_cubic(t)
        
        canvas = Image.new("RGB", (w, h))
        blend_width = max(10, min(w, h) // 20)  # 混合边缘宽度
        
        if direction in ("left", "right"):
            offset = int(w * t_smooth)
            if direction == "left":
                # B从右往左推进，A往左退出
                if offset > 0:
                    # B的右侧部分
                    B_part = B.crop((w - offset, 0, w, h))
                    if offset < w:
                        B_part = B_part.resize((offset, h), Image.LANCZOS)
                    canvas.paste(B_part, (0, 0))
                
                # A的剩余部分
                if offset < w:
                    A_part = A.crop((0, 0, w - offset, h))
                    canvas.paste(A_part, (offset, 0))
                    
                    # 在边界处添加混合效果
                    if offset > blend_width and offset < w:
                        blend_mask = Image.new("L", (blend_width, h))
                        blend_draw = ImageDraw.Draw(blend_mask)
                        for x in range(blend_width):
                            alpha = int(255 * x / blend_width)
                            blend_draw.rectangle([x, 0, x + 1, h], fill=alpha)
                        blend_region = Image.composite(
                            B.crop((w - offset, 0, w - offset + blend_width, h)),
                            A.crop((offset - blend_width, 0, offset, h)),
                            blend_mask
                        )
                        canvas.paste(blend_region, (offset - blend_width, 0))
            else:  # right
                if offset > 0:
                    B_part = B.crop((0, 0, offset, h))
                    if offset < w:
                        B_part = B_part.resize((offset, h), Image.LANCZOS)
                    canvas.paste(B_part, (w - offset, 0))
                
                if offset < w:
                    A_part = A.crop((offset, 0, w, h))
                    canvas.paste(A_part, (0, 0))
                    
                    if offset > blend_width and offset < w:
                        blend_mask = Image.new("L", (blend_width, h))
                        blend_draw = ImageDraw.Draw(blend_mask)
                        for x in range(blend_width):
                            alpha = int(255 * (blend_width - x) / blend_width)
                            blend_draw.rectangle([x, 0, x + 1, h], fill=alpha)
                        blend_region = Image.composite(
                            A.crop((offset - blend_width, 0, offset, h)),
                            B.crop((0, 0, blend_width, h)),
                            blend_mask
                        )
                        canvas.paste(blend_region, (offset - blend_width, 0))
        else:
            offset = int(h * t_smooth)
            if direction == "up":
                if offset > 0:
                    B_part = B.crop((0, h - offset, w, h))
                    if offset < h:
                        B_part = B_part.resize((w, offset), Image.LANCZOS)
                    canvas.paste(B_part, (0, 0))
                
                if offset < h:
                    A_part = A.crop((0, 0, w, h - offset))
                    canvas.paste(A_part, (0, offset))
                    
                    if offset > blend_width and offset < h:
                        blend_mask = Image.new("L", (w, blend_width))
                        blend_draw = ImageDraw.Draw(blend_mask)
                        for y in range(blend_width):
                            alpha = int(255 * y / blend_width)
                            blend_draw.rectangle([0, y, w, y + 1], fill=alpha)
                        blend_region = Image.composite(
                            B.crop((0, h - offset, w, h - offset + blend_width)),
                            A.crop((0, offset - blend_width, w, offset)),
                            blend_mask
                        )
                        canvas.paste(blend_region, (0, offset - blend_width))
            else:  # down
                if offset > 0:
                    B_part = B.crop((0, 0, w, offset))
                    if offset < h:
                        B_part = B_part.resize((w, offset), Image.LANCZOS)
                    canvas.paste(B_part, (0, h - offset))
                
                if offset < h:
                    A_part = A.crop((0, offset, w, h))
                    canvas.paste(A_part, (0, 0))
                    
                    if offset > blend_width and offset < h:
                        blend_mask = Image.new("L", (w, blend_width))
                        blend_draw = ImageDraw.Draw(blend_mask)
                        for y in range(blend_width):
                            alpha = int(255 * (blend_width - y) / blend_width)
                            blend_draw.rectangle([0, y, w, y + 1], fill=alpha)
                        blend_region = Image.composite(
                            A.crop((0, offset - blend_width, w, offset)),
                            B.crop((0, 0, w, blend_width)),
                            blend_mask
                        )
                        canvas.paste(blend_region, (0, offset - blend_width))
        return canvas

    # ----------------------------
    # 过渡：zoom（优化版 - 使用缓动和更平滑的缩放）
    # ----------------------------
    def _zoom(self, A, B, t, mode):
        w, h = A.size
        # 使用缓动函数让缩放更自然
        t_smooth = self._ease_out_cubic(t)
        
        if mode == "in":
            # B 从 70% 缩放到 100%，使用更平滑的曲线
            scale_start = 0.7
            scale = scale_start + (1.0 - scale_start) * t_smooth
            
            # 同时进行淡入效果
            blend_t = self._smooth_step(t)
            
            Bw = max(1, int(w * scale))
            Bh = max(1, int(h * scale))
            B_resized = B.resize((Bw, Bh), Image.LANCZOS)
            
            # 创建带透明度的混合
            canvas = A.copy()
            paste_x = (w - Bw) // 2
            paste_y = (h - Bh) // 2
            
            # 使用混合而不是直接粘贴，让过渡更平滑
            B_layer = Image.new("RGB", (w, h))
            B_layer.paste(B_resized, (paste_x, paste_y))
            return Image.blend(A, B_layer, blend_t)
        else:
            # zoom_out: A 缩小并混合 B，使用更平滑的曲线
            scale_start = 1.0
            scale_end = 0.6
            scale = scale_start - (scale_start - scale_end) * t_smooth
            
            blend_t = self._smooth_step(t)
            
            Aw = max(1, int(w * scale))
            Ah = max(1, int(h * scale))
            A_resized = A.resize((Aw, Ah), Image.LANCZOS)
            
            # 将缩小的A放在B上，然后混合
            A_layer = Image.new("RGB", (w, h))
            paste_x = (w - Aw) // 2
            paste_y = (h - Ah) // 2
            A_layer.paste(A_resized, (paste_x, paste_y))
            
            return Image.blend(A_layer, B, blend_t)

    # ----------------------------
    # 过渡：轻微模糊以软化衔接（优化版）
    # ----------------------------
    def _soft_blur_blend(self, A, B, t):
        # 使用缓动函数
        t_smooth = self._ease_in_out_quad(t)
        
        # 给 A 应用逐步增强的模糊，然后 blend 到 B，效果柔和
        # 使用更平滑的模糊曲线
        max_blur = 15
        radius = 1 + int(max_blur * t_smooth * t_smooth)  # 二次曲线让模糊更自然
        
        Ab = A.filter(ImageFilter.GaussianBlur(radius=min(radius, max_blur)))
        
        # 同时给B也添加轻微模糊，让过渡更平滑
        if t_smooth > 0.3:
            B_blur_radius = int(3 * (t_smooth - 0.3) / 0.7)
            if B_blur_radius > 0:
                B = B.filter(ImageFilter.GaussianBlur(radius=min(B_blur_radius, 5)))
        
        return Image.blend(Ab, B, t_smooth)

    # ----------------------------
    # 过渡：旋转过渡（新增 - 让过渡更丰富）
    # ----------------------------
    def _rotate_transition(self, A, B, t, direction="clockwise"):
        w, h = A.size
        # 使用缓动函数
        t_smooth = self._ease_in_out_cubic(t)
        
        # 旋转角度：从0度到45度（更温和的旋转）
        max_angle = 45
        if direction == "clockwise":
            angle = max_angle * t_smooth
        else:  # counterclockwise
            angle = -max_angle * t_smooth
        
        # A旋转并淡出
        A_rotated = A.rotate(angle, expand=False, resample=Image.BICUBIC, fillcolor=(0, 0, 0))
        A_alpha = int(255 * (1 - t_smooth))
        
        # B旋转并淡入（反向旋转）
        B_angle = -angle if direction == "clockwise" else angle
        B_rotated = B.rotate(B_angle, expand=False, resample=Image.BICUBIC, fillcolor=(0, 0, 0))
        B_alpha = int(255 * t_smooth)
        
        # 创建带透明度的图层
        A_rgba = A_rotated.convert("RGBA")
        A_alpha_channel = Image.new("L", (w, h), A_alpha)
        A_layer = Image.merge("RGBA", (*A_rgba.split()[:3], A_alpha_channel))
        
        B_rgba = B_rotated.convert("RGBA")
        B_alpha_channel = Image.new("L", (w, h), B_alpha)
        B_layer = Image.merge("RGBA", (*B_rgba.split()[:3], B_alpha_channel))
        
        # 合成
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 255))
        canvas = Image.alpha_composite(canvas, A_layer)
        canvas = Image.alpha_composite(canvas, B_layer)
        
        return canvas.convert("RGB")

    # ----------------------------
    # 统一的过渡入口（返回 PIL.Image）
    # ----------------------------
    def apply_transition_frame(self, pathA, pathB, t, transition_type):
        A, B = self._open_and_match(pathA, pathB)

        # 若为 random，则在具体候选中随机选择
        if transition_type == "random":
            candidates = [
                "crossfade", "left_wipe", "right_wipe", #"up_wipe", "down_wipe",
                "slide_left", "slide_right", #"slide_up", "slide_down",
                "zoom_in", "zoom_out", "soft_blur"
            ]
            transition_type = random.choice(candidates)

        if transition_type == "crossfade":
            return self._crossfade(A, B, t)
        if transition_type in ("left_wipe", "right_wipe", "up_wipe", "down_wipe"):
            dir_map = {
                "left_wipe": "left", "right_wipe": "right",
                "up_wipe": "up", "down_wipe": "down"
            }
            return self._wipe(A, B, t, dir_map[transition_type])
        if transition_type.startswith("slide_"):
            dir_map = {
                "slide_left": "left", "slide_right": "right",
                "slide_up": "up", "slide_down": "down"
            }
            return self._slide(A, B, t, dir_map[transition_type])
        if transition_type == "zoom_in":
            return self._zoom(A, B, t, "in")
        if transition_type == "zoom_out":
            return self._zoom(A, B, t, "out")
        if transition_type == "soft_blur":
            return self._soft_blur_blend(A, B, t)

        # fallback
        return self._crossfade(A, B, t)

    # ----------------------------
    # 音频相关（与原逻辑一致）
    # ----------------------------
    def get_audio_duration(self, audio_path):
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out = result.stdout.strip()
        try:
            return float(out)
        except:
            return 0.0

    def save_audio_to_temp_file(self, audio_input):
        waveform = audio_input["waveform"]
        sample_rate = audio_input["sample_rate"]

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu()
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        torchaudio.save(tmp_path, waveform, sample_rate)
        return tmp_path

    # ----------------------------
    # 主逻辑：执行合成
    # ----------------------------
    def execute_video_synth(
        self,
        image_paths,
        frame_counts,
        audio,
        fps,
        filename_prefix,
        format="mp4",
        enable_transition=True,
        transition_seconds=0.5,
        transition_type="crossfade",
    ):
        # 1. 保存音频
        audio_temp_path = self.save_audio_to_temp_file(audio)

        # 2. 解析路径和帧数
        paths_list = [p.strip() for p in image_paths.split("\n") if p.strip()]
        counts_list = [int(c.strip()) for c in frame_counts.split("\n") if c.strip()]

        if len(paths_list) != len(counts_list):
            raise ValueError("路径数量与帧数数量不一致")

        all_frames = []
        folder_start_indices = []
        current_index = 0

        for path, count in zip(paths_list, counts_list):
            path = path.strip('"').strip("'")
            folder_start_indices.append(current_index)

            if os.path.isdir(path):
                valid_exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')
                images = []
                for ext in valid_exts:
                    images.extend(glob.glob(os.path.join(path, ext)))
                images.sort()

                if images:
                    selected = images[:count]
                    if len(selected) < count:
                        selected.extend([selected[-1]] * (count - len(selected)))
                    all_frames.extend(selected)
                else:
                    # 文件夹为空就跳过（不会增加索引）
                    continue
            else:
                # 单文件重复 count 次
                all_frames.extend([path] * count)

            current_index += count

        total_frames = len(all_frames)
        if total_frames == 0:
            raise ValueError("错误: 未收集到任何图片帧。")

        print(f"收集到总帧数: {total_frames}")

        # 3. 过渡处理（替换边界帧，保持总帧数不变）
        # 收集所有过渡帧临时文件路径，用于后续清理
        transition_temp_files = []
        
        if enable_transition and len(folder_start_indices) > 1:
            # 过渡帧数（至少 1）
            fade_len = max(1, int(round(fps * transition_seconds)))
            print(f"[Transition] 启用，类型: {transition_type}, 秒: {transition_seconds}, 帧数: {fade_len}")

            # 注意：我们将用过渡帧替换边界处的原始帧，而不是插入新帧
            # 这样可以保持总帧数不变
            # 原始文件路径保持不变，过渡帧使用临时文件
            for i in range(len(folder_start_indices) - 1):
                A_start = folder_start_indices[i]
                B_start = folder_start_indices[i + 1]
                A_end = B_start - 1

                # 计算两侧可用帧数（保证过渡不超边界）
                available_A = A_end - A_start + 1
                next_folder_start = folder_start_indices[i + 2] if (i + 2) < len(folder_start_indices) else total_frames
                available_B = next_folder_start - B_start

                # 先确定实际可替换的帧数
                replace_from_A = min(fade_len // 2, available_A)
                replace_from_B = min(fade_len - fade_len // 2, available_B)
                actual_replace = replace_from_A + replace_from_B
                
                if actual_replace <= 1:
                    continue

                # 若是 random：为当前段随机选择一个过渡类型（更自然）
                chosen_type = transition_type
                if transition_type == "random":
                    opts = [
                        "crossfade", "left_wipe", "right_wipe", "up_wipe", "down_wipe",
                        "slide_left", "slide_right", "slide_up", "slide_down",
                        "zoom_in", "zoom_out", "soft_blur"
                    ]
                    chosen_type = random.choice(opts)

                print(f"[Transition] 段 {i}→{i+1} | 类型: {chosen_type} | 使用帧数: {actual_replace} (A:{replace_from_A}, B:{replace_from_B})")

                # 生成 transition 帧
                # 关键优化：在过渡过程中，B画面应该使用B段中对应位置的帧，而不是只用B的第一帧
                # 这样B画面在过渡过程中会动态变化，而不是静态的
                # 例如在"向左推"效果中，右侧的B画面会随着时间播放，而不是只显示B的第一帧
                transition_paths = []
                for k in range(actual_replace):
                    # t值从0到1均匀分布，确保整个过渡过程都有变化
                    t = k / (actual_replace - 1) if actual_replace > 1 else 0.0
                    
                    # 根据过渡进度，选择A段和B段中对应位置的帧
                    # A段：从末尾往前取，让A画面在过渡过程中也动态变化
                    # 让A画面从A段的末尾开始，随着过渡进度逐渐使用A段更早的帧
                    # 这样在"向左推"等效果中，左侧的A画面会随着时间播放，而不是静态的
                    A_frame_idx = A_end - k
                    # 确保不超出A段范围
                    A_frame_idx = max(A_frame_idx, A_start)
                    
                    # B段：从开始往后取，让B画面在整个过渡过程中动态变化
                    # 让B画面从B的第一帧开始，随着过渡进度逐渐使用B段后续的帧
                    # 这样在"向左推"等效果中，右侧的B画面会随着时间播放，而不是静态的
                    B_frame_idx = B_start + k
                    # 确保不超出B段范围
                    B_frame_idx = min(B_frame_idx, B_start + available_B - 1)
                    
                    A_path = all_frames[A_frame_idx]
                    B_path = all_frames[B_frame_idx]
                    
                    # 读取原始文件生成过渡帧（不修改原始文件）
                    img = self.apply_transition_frame(A_path, B_path, t, chosen_type)
                    # 将过渡帧保存到临时文件（不会影响原始文件）
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    tmp_path = tmp.name
                    tmp.close()
                    img.save(tmp_path)
                    transition_paths.append(tmp_path)
                    # 记录临时文件路径，用于后续清理
                    transition_temp_files.append(tmp_path)
                
                # 替换 A 段最后的帧（使用过渡帧的前半部分，从 A 到中间）
                replace_A_start = A_end - replace_from_A + 1
                for idx in range(replace_from_A):
                    # 使用对应的过渡帧索引，确保 t 值从 0 开始递增
                    trans_path = transition_paths[idx]
                    all_frames[replace_A_start + idx] = trans_path
                
                # 替换 B 段前面的帧（使用过渡帧的后半部分，从中间到 B）
                for idx in range(replace_from_B):
                    # 使用过渡帧的后半部分，索引从 replace_from_A 开始
                    trans_path = transition_paths[replace_from_A + idx]
                    all_frames[B_start + idx] = trans_path
                
                # 关键修复：删除B段中已经被过渡帧替换的原始帧，避免重复
                # 由于过渡帧已经替换了B段的前 replace_from_B 帧
                # 需要从 all_frames 中删除B段中从 replace_from_B 开始的原始帧
                # 删除从 B_start + replace_from_B 开始的 replace_from_B 个帧
                del_start = B_start + replace_from_B
                del_end = del_start + replace_from_B
                if del_end <= len(all_frames):
                    # 删除B段中已经被替换的原始帧
                    del all_frames[del_start:del_end]
                    # 更新总帧数
                    total_frames -= replace_from_B
                    # 更新后续段的起始索引
                    for j in range(i + 2, len(folder_start_indices)):
                        folder_start_indices[j] -= replace_from_B
                
                print(f"[Transition] 替换: A段最后{replace_from_A}帧, B段前{replace_from_B}帧, 删除B段重复帧{replace_from_B}帧, 总帧数: {total_frames}")

        else:
            print("[Transition] 未启用或无多段，跳过过渡处理")

        # 4. 将帧复制到临时目录并用 ffmpeg 合成（与之前逻辑一致）
        temp_dir = os.path.join(folder_paths.get_temp_directory(), "debug_frames_video_synth")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        for i, img_path in enumerate(all_frames):
            shutil.copy(img_path, os.path.join(temp_dir, f"{i:05d}.png"))

        # 计算 input_fps，使得帧按音频时长播放
        audio_duration = self.get_audio_duration(audio_temp_path)
        if audio_duration <= 0:
            raise ValueError("错误: 音频时长无效。")
        input_fps = len(all_frames) / audio_duration

        print(f"最终帧数: {len(all_frames)}, 音频时长: {audio_duration}s, input_fps: {input_fps:.4f}")

        output_dir = folder_paths.get_output_directory()
        output_filename = f"{filename_prefix}.{format}"
        output_file_path = os.path.join(output_dir, output_filename)
        counter = 1
        while os.path.exists(output_file_path):
            output_filename = f"{filename_prefix}_{counter}.{format}"
            output_file_path = os.path.join(output_dir, output_filename)
            counter += 1

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(input_fps),
            "-i", os.path.join(temp_dir, "%05d.png"),
            "-i", audio_temp_path,
            "-r", str(fps),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-shortest",
            output_file_path
        ]

        subprocess.run(ffmpeg_cmd, check=True)

        # 清理临时文件
        # 清理临时音频
        try:
            if os.path.exists(audio_temp_path):
                os.remove(audio_temp_path)
        except Exception:
            pass
        
        # 清理过渡帧临时文件（确保不留下垃圾文件，原始文件路径保持不变）
        for temp_file in transition_temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:
                pass

        print(f"视频生成成功: {output_file_path}")
        return (output_file_path,)
    


class WanVideoImageToVideoInfiniteTalkFunCamera:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Width of the generation"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 29048, "step": 8, "tooltip": "Height of the generation"}),
            "frame_window_size": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "The number of frames to process at once, should be a value the model is generally good at."}),
            "motion_frame": ("INT", {"default": 25, "min": 1, "max": 10000, "step": 1, "tooltip": "Driven frame length used in the long video generation. Basically the overlap length."}),
            "force_offload": ("BOOLEAN", {"default": False, "tooltip": "Whether to force offload the model within the loop for VAE operations, enable if you encounter memory issues."}),
            "colormatch": (
            [   
                'disabled',
                'mkl',
                'hm', 
                'reinhard', 
                'mvgd', 
                'hm-mvgd-hm', 
                'hm-mkl-hm',
            ], {
               "default": 'disabled', "tooltip": "Color matching method to use between the windows"
            },),
            },
            "optional": {
                "start_image": ("IMAGE", {"tooltip": "Images to encode"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "control_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Control signal for the Fun -model"}),
                "fun_or_fl2v_model": ("BOOLEAN", {"default": True, "tooltip": "Enable when using official FLF2V or Fun model"}),
                "mode": ([
                    "auto",
                    "multitalk",
                    "infinitetalk"
                ], {"default": "auto", "tooltip": "The sampling strategy to use in the long video generation loop, should match the model used"}),
                "output_path": ("STRING", {"default": "", "tooltip": "If set, will save each window's resulting frames to this folder, also DISABLES returning the final video tensor to save memory"}),

            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "STRING",)
    RETURN_NAMES = ("image_embeds", "output_path")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Enables Multi/InfiniteTalk long video generation sampling method, the video is created in windows with overlapping frames. Not compatible or necessary to be used with context windows and many other features besides Multi/InfiniteTalk."

    def process(self, vae, width, height, frame_window_size, motion_frame, force_offload, colormatch, start_image=None, tiled_vae=False, clip_embeds=None, control_embeds = None, fun_or_fl2v_model=False, mode="multitalk", output_path=""):

        H = height
        W = width
        VAE_STRIDE = (4, 8, 8)
        
        num_frames = ((frame_window_size - 1) // 4) * 4 + 1

        # Resize and rearrange the input image dimensions
        if start_image is not None:
            resized_start_image = common_upscale(start_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            resized_start_image = resized_start_image * 2 - 1
            resized_start_image = resized_start_image.unsqueeze(0)
        
        target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                        height // VAE_STRIDE[1],
                        width // VAE_STRIDE[2])
        
        if output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_path, f"{timestamp}_{mode}_output")
            os.makedirs(output_path, exist_ok=True)

        image_embeds = {
            "multitalk_sampling": True,
            "multitalk_start_image": resized_start_image if start_image is not None else None,
            "frame_window_size": num_frames,
            "motion_frame": motion_frame,
            "target_h": H,
            "target_w": W,
            "lat_h": H,
            "lat_w": W,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
            "tiled_vae": tiled_vae,
            "force_offload": force_offload,
            "fun_or_fl2v_model": fun_or_fl2v_model,
            "vae": vae,
            "target_shape": target_shape,
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "colormatch": colormatch,
            "multitalk_mode": mode,
            "output_path": output_path
        }

        return (image_embeds, output_path)
    

from comfy import model_management as mm
import os, gc, math
from .utils import(log, clip_encode_image_tiled, add_noise_to_reference_video, set_module_tensor_to_device)

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

class WanVideoImageToVideoEncodeForInfiniteTalkFunCamera:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Strength of noise augmentation, helpful for I2V where some noise can add motion and give sharper results"}),
            "start_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
            "end_latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("WANVAE",),
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "start_image": ("IMAGE", {"tooltip": "Image to encode"}),
                "end_image": ("IMAGE", {"tooltip": "end frame"}),
                "control_embeds": ("WANVIDIMAGE_EMBEDS", {"tooltip": "Control signal for the Fun -model"}),
                "fun_or_fl2v_model": ("BOOLEAN", {"default": True, "tooltip": "Enable when using official FLF2V or Fun model"}),
                "temporal_mask": ("MASK", {"tooltip": "mask"}),
                "extra_latents": ("LATENT", {"tooltip": "Extra latents to add to the input front, used for Skyreels A2 reference images"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "add_cond_latents": ("ADD_COND_LATENTS", {"advanced": True, "tooltip": "Additional cond latents WIP"}),
                "output_path": ("STRING", {"default": "", "tooltip": "If set, will save each window's resulting frames to this folder, also DISABLES returning the final video tensor to save memory"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(self, width, height, num_frames, force_offload, noise_aug_strength, 
                start_latent_strength, end_latent_strength, start_image=None, end_image=None, control_embeds=None, fun_or_fl2v_model=False, 
                temporal_mask=None, extra_latents=None, clip_embeds=None, tiled_vae=False, add_cond_latents=None, vae=None, output_path=""):
        
        if vae is None:
            raise ValueError("VAE is required for image encoding.")
        H = height
        W = width
           
        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        num_frames = ((num_frames - 1) // 4) * 4 + 1
        two_ref_images = start_image is not None and end_image is not None

        if start_image is None and end_image is not None:
            fun_or_fl2v_model = True # end image alone only works with this option

        base_frames = num_frames + (1 if two_ref_images and not fun_or_fl2v_model else 0)
        if temporal_mask is None:
            mask = torch.zeros(1, base_frames, lat_h, lat_w, device=device, dtype=vae.dtype)
            if start_image is not None:
                mask[:, 0:start_image.shape[0]] = 1  # First frame
            if end_image is not None:
                mask[:, -end_image.shape[0]:] = 1  # End frame if exists
        else:
            mask = common_upscale(temporal_mask.unsqueeze(1).to(device), lat_w, lat_h, "nearest", "disabled").squeeze(1)
            if mask.shape[0] > base_frames:
                mask = mask[:base_frames]
            elif mask.shape[0] < base_frames:
                mask = torch.cat([mask, torch.zeros(base_frames - mask.shape[0], lat_h, lat_w, device=device)])
            mask = mask.unsqueeze(0).to(device, vae.dtype)

        # Repeat first frame and optionally end frame
        start_mask_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1) # T, C, H, W
        if end_image is not None and not fun_or_fl2v_model:
            end_mask_repeated = torch.repeat_interleave(mask[:, -1:], repeats=4, dim=1) # T, C, H, W
            mask = torch.cat([start_mask_repeated, mask[:, 1:-1], end_mask_repeated], dim=1)
        else:
            mask = torch.cat([start_mask_repeated, mask[:, 1:]], dim=1)

        # Reshape mask into groups of 4 frames
        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w) # 1, T, C, H, W
        mask = mask.movedim(1, 2)[0]# C, T, H, W

        # Resize and rearrange the input image dimensions
        if start_image is not None:
            start_image = start_image[..., :3]
            if start_image.shape[1] != H or start_image.shape[2] != W:
                resized_start_image = common_upscale(start_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_start_image = start_image.permute(3, 0, 1, 2) # C, T, H, W
            resized_start_image = resized_start_image * 2 - 1
            if noise_aug_strength > 0.0:
                resized_start_image = add_noise_to_reference_video(resized_start_image, ratio=noise_aug_strength)
        
        if end_image is not None:
            end_image = end_image[..., :3]
            if end_image.shape[1] != H or end_image.shape[2] != W:
                resized_end_image = common_upscale(end_image.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_end_image = end_image.permute(3, 0, 1, 2) # C, T, H, W
            resized_end_image = resized_end_image * 2 - 1
            if noise_aug_strength > 0.0:
                resized_end_image = add_noise_to_reference_video(resized_end_image, ratio=noise_aug_strength)
            
        # Concatenate image with zero frames and encode
        if temporal_mask is None:
            if start_image is not None and end_image is None:
                zero_frames = torch.zeros(3, num_frames-start_image.shape[0], H, W, device=device, dtype=vae.dtype)
                concatenated = torch.cat([resized_start_image.to(device, dtype=vae.dtype), zero_frames], dim=1)
                del resized_start_image, zero_frames
            elif start_image is None and end_image is not None:
                zero_frames = torch.zeros(3, num_frames-end_image.shape[0], H, W, device=device, dtype=vae.dtype)
                concatenated = torch.cat([zero_frames, resized_end_image.to(device, dtype=vae.dtype)], dim=1)
                del zero_frames
            elif start_image is None and end_image is None:
                concatenated = torch.zeros(3, num_frames, H, W, device=device, dtype=vae.dtype)
            else:
                if fun_or_fl2v_model:
                    zero_frames = torch.zeros(3, num_frames-(start_image.shape[0]+end_image.shape[0]), H, W, device=device, dtype=vae.dtype)
                else:
                    zero_frames = torch.zeros(3, num_frames-1, H, W, device=device, dtype=vae.dtype)
                concatenated = torch.cat([resized_start_image.to(device, dtype=vae.dtype), zero_frames, resized_end_image.to(device, dtype=vae.dtype)], dim=1)
                del resized_start_image, zero_frames
        else:
            temporal_mask = common_upscale(temporal_mask.unsqueeze(1), W, H, "nearest", "disabled").squeeze(1)
            concatenated = resized_start_image[:,:num_frames].to(vae.dtype)# * temporal_mask[:num_frames].unsqueeze(0).to(vae.dtype)
            del resized_start_image, temporal_mask

        mm.soft_empty_cache()
        gc.collect()

        vae.to(device)
        y = vae.encode([concatenated], device, end_=(end_image is not None and not fun_or_fl2v_model),tiled=tiled_vae)[0]
        del concatenated

        has_ref = False
        if extra_latents is not None:
            samples = extra_latents["samples"].squeeze(0)
            y = torch.cat([samples, y], dim=1)
            mask = torch.cat([torch.ones_like(mask[:, 0:samples.shape[1]]), mask], dim=1)
            num_frames += samples.shape[1] * 4
            has_ref = True
        y[:, :1] *= start_latent_strength
        y[:, -1:] *= end_latent_strength

        # Calculate maximum sequence length
        patches_per_frame = lat_h * lat_w // (PATCH_SIZE[1] * PATCH_SIZE[2])
        frames_per_stride = (num_frames - 1) // 4 + (2 if end_image is not None and not fun_or_fl2v_model else 1)
        max_seq_len = frames_per_stride * patches_per_frame

        if add_cond_latents is not None:
            add_cond_latents["ref_latent_neg"] = vae.encode(torch.zeros(1, 3, 1, H, W, device=device, dtype=vae.dtype), device)
        
        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()
        if output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_path, f"{timestamp}_infinitetalk_output")
            os.makedirs(output_path, exist_ok=True)
        image_embeds = {
            "image_embeds": y.cpu(),
            "clip_context": clip_embeds.get("clip_embeds", None) if clip_embeds is not None else None,
            "negative_clip_context": clip_embeds.get("negative_clip_embeds", None) if clip_embeds is not None else None,
            "max_seq_len": max_seq_len,
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
            "control_embeds": control_embeds["control_embeds"] if control_embeds is not None else None,
            "end_image": resized_end_image if end_image is not None else None,
            "fun_or_fl2v_model": fun_or_fl2v_model,
            "has_ref": has_ref,
            "add_cond_latents": add_cond_latents,
            "mask": mask.cpu(),
            "output_path": output_path
        }

        return (image_embeds,)
        

NODE_CLASS_MAPPINGS = {
    "InfiniteTalkMultiImage": InfiniteTalkMultiImage,
    "WanVideoImageToVideoInfiniteTalkFunCamera": WanVideoImageToVideoInfiniteTalkFunCamera,
    "WanVideoImageToVideoEncodeForInfiniteTalkFunCamera": WanVideoImageToVideoEncodeForInfiniteTalkFunCamera,

    "MakeBatchFromIntList": MakeBatchFromIntList,
    "GetIntByIndex": GetIntByIndex,
    "GetFloatByIndex": GetFloatByIndex,
    "MakeBatchFromFloatList": MakeBatchFromFloatList,
    "InfiniteTalkEmbedsSlice": InfiniteTalkEmbedsSlice,
    "AudioSmartSlice": AudioSmartSlice,

    "VideoFromPathsAndAudio": VideoFromPathsAndAudio

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InfiniteTalkMultiImage": "InfiniteTalkMultiImage",
    "WanVideoImageToVideoEncodeForInfiniteTalkFunCamera": "WanVideoImageToVideoEncodeForInfiniteTalkFunCamera",
    "MakeBatchFromIntList": "MakeBatchFromIntList",
    "GetIntByIndex": "GetIntByIndex",
    "GetFloatByIndex": "GetFloatByIndex",
    "MakeBatchFromFloatList": "MakeBatchFromFloatList",
    "InfiniteTalkEmbedsSlice": "InfiniteTalkEmbedsSlice",
    "AudioSmartSlice": "AudioSmartSlice",
    
    "VideoFromPathsAndAudio": "Video Synth (Path List + Audio)"

}