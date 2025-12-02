

import torchaudio
import torch
import numpy as np
from tqdm import tqdm
import webrtcvad



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
        step = Decimal("0.04")

        auto_start_time_list = kwargs.get("auto_start_time_list", [])

        print("auto_start_time_list:", auto_start_time_list)

        # =============== 音频长度 Decimal 无损读取 ===============
        audio_duration = Decimal(str(kwargs.get("audio_duration", "0")))
        audio_duration = align_step_004(audio_duration)

        # =============== 文本 ===============
        raw_text = kwargs.get("prompt_list_input", "")
        text_list = [line.strip() for line in raw_text.split("\n") if line.strip()]

        # =============== 图片 + start_time ===============
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

        start_times = [p[0] for p in pairs]  # Decimal
        images = [p[1] for p in pairs]

        if auto_start_time_list is not None and len(auto_start_time_list) > 0:
            start_times = [Decimal(str(t)) for t in auto_start_time_list]

        min_len = min(len(images), len(start_times))

        print("min_len", min_len)
        
        images = images[:min_len]
        start_times = start_times[:min_len]

        # =============== frame count（真实差值） ===============
        frame_count_list = []

        for idx, st in enumerate(start_times):
            if idx < len(start_times) - 1:
                end = start_times[idx + 1]
            else:
                end = audio_duration

            dur = end - st
            if dur < 0:
                dur = Decimal("0")

            frames = int(dur * fps)  # Decimal → int 无损
            frame_count_list.append(frames)

        # =============== real_start_time_list ===============
        real_start_time_list = [float(t) for t in start_times]

        # =============== prompt 对齐 ===============
        n_images = len(images)
        n_texts = len(text_list)

        if n_texts < n_images:
            # 如果文本数量不够，用空字符串填充
            text_list = text_list + [""] * (n_images - n_texts)
        elif n_texts > n_images:
            # 如果文本数量多于图片数量，截断多余部分
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
import shutil
import subprocess
import tempfile
import folder_paths
import sys
import glob

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
                # 新增 fps 参数，默认 25
                "fps": ("INT", {"default": 25, "min": 1, "max": 120, "step": 1, "display": "number"}),
                "filename_prefix": ("STRING", {"default": "video_output"}),
                "format": (["mp4", "mkv", "mov"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "execute_video_synth"
    CATEGORY = "Custom/Video"

    def get_audio_duration(self, audio_path):
        """获取音频时长(秒)"""
        try:
            cmd = [
                "ffprobe", 
                "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                audio_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            duration = float(result.stdout.strip())
            return duration
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return 0.0
        
    def save_audio_to_temp_file(self, audio_input):
        """
        将 ComfyUI 的 AUDIO 对象 ({"waveform": tensor, "sample_rate": int}) 
        保存为临时的 .wav 文件供 ffmpeg 使用
        """
        # 解析 AUDIO 对象结构
        waveform = audio_input['waveform'] # Shape通常是 [batch, channels, time] 或 [channels, time]
        sample_rate = audio_input['sample_rate']

        # 确保 waveform 是 CPU tensor
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu()
        
        # 处理维度问题：torchaudio.save 期望 (channels, time)
        # 如果是 3维 (1, C, T)，需要 squeeze
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        
        # 创建临时文件
        # delete=False 因为我们需要关闭文件后让 ffmpeg 读取它
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close() # 关闭句柄，让 torchaudio 写入

        try:
            torchaudio.save(temp_audio_path, waveform, sample_rate)
            print(f"音频对象已缓存至: {temp_audio_path}")
            return temp_audio_path
        except Exception as e:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            raise ValueError(f"保存临时音频文件失败: {e}")
        
    def execute_video_synth(self, image_paths, frame_counts, audio, fps, filename_prefix, format="mp4"):
        # 0. 预处理音频：将 AUDIO 对象转为临时文件路径
        # audio 参数现在是来自 LoadAudioUpload 的字典
        audio_temp_path = self.save_audio_to_temp_file(audio)

        # 1. 解析输入路径和帧数
        paths_list = [p.strip() for p in image_paths.split('\n') if p.strip()]
        counts_list = [int(c.strip()) for c in frame_counts.split('\n') if c.strip()]

        if len(paths_list) != len(counts_list):
            raise ValueError(f"错误: 路径数量 ({len(paths_list)}) 与 帧数设定数量 ({len(counts_list)}) 不一致。")


        # 2. 收集所有图片帧
        all_frames = []
        print(f"开始处理 {len(paths_list)} 组输入...")
        
        for path, count in zip(paths_list, counts_list):
            # 去除可能存在的引号
            path = path.strip('"').strip("'")
            
            if not os.path.exists(path):
                print(f"警告: 路径不存在，跳过: {path}")
                continue

            if os.path.isdir(path):
                # 读取文件夹内图片
                valid_exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')
                images = []
                for ext in valid_exts:
                    images.extend(glob.glob(os.path.join(path, ext)))
                
                # 按文件名排序确保顺序正确
                images.sort()
                
                if images:
                    # 截取前 count 张
                    selected = images[:count]
                    # 如果文件夹里的图不够 count 数量，用最后一张图补齐
                    if len(selected) < count:
                        diff = count - len(selected)
                        selected.extend([selected[-1]] * diff)
                    
                    all_frames.extend(selected)
            
            elif os.path.isfile(path):
                # 单张图片重复 count 次
                all_frames.extend([path] * count)

        total_frames = len(all_frames)
        if total_frames == 0:
            raise ValueError("错误: 未收集到任何图片帧。")

        # 3. 计算同步所需的输入帧率 (Input FPS)
        audio_duration = self.get_audio_duration(audio_temp_path)
        if audio_duration <= 0:
            raise ValueError("错误: 音频时长无效。")
        
        # 计算逻辑：为了让 all_frames 刚好播完等于 audio_duration，输入流速度必须是 input_fps
        input_fps = total_frames / audio_duration
        print(f"统计: 总帧数 {total_frames} | 音频时长 {audio_duration}s | 计算输入流速: {input_fps:.4f} fps")
        print(f"设置: 目标输出视频帧率: {fps} fps")

        # 4. 准备输出文件
        output_dir = folder_paths.get_output_directory()
        output_filename = f"{filename_prefix}.{format}"
        output_file_path = os.path.join(output_dir, output_filename)
        
        # 避免文件名冲突
        counter = 1
        while os.path.exists(output_file_path):
            output_filename = f"{filename_prefix}_{counter}.{format}"
            output_file_path = os.path.join(output_dir, output_filename)
            counter += 1

        # 5. 临时目录处理与合成
        # 1. 不使用临时目录，而是使用 ComfyUI 根目录下的 temp/debug_frames 文件夹
        # 获取 ComfyUI 的 temp 目录
        comfy_temp_dir = folder_paths.get_temp_directory()
        # 创建一个固定的子文件夹，每次运行前清空（或者不清空看你需要）
        temp_dir = os.path.join(comfy_temp_dir, "debug_frames_video_synth")
        
        # 如果目录存在，先清空旧文件（防止混淆），如果不存在则创建
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        print(f"★ 中间帧图片已保存在: {temp_dir}")

        # 去掉 'with tempfile...' 的缩进，直接执行
        try:
            print("正在准备帧序列...")
            for i, img_path in enumerate(all_frames):
                temp_img_name = f"{i:05d}.png"
                shutil.copy(img_path, os.path.join(temp_dir, temp_img_name))

            print("开始执行 FFmpeg 合成...")
            
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
            
        except Exception as e:
            raise e
        finally:
            # 6. 清理临时音频文件
            # 无论成功还是失败，都要把 save_audio_to_temp_file 生成的文件删掉
            if audio_temp_path and os.path.exists(audio_temp_path):
                try:
                    os.remove(audio_temp_path)
                    print(f"已清理临时音频文件: {audio_temp_path}")
                except Exception as e:
                    print(f"清理临时音频文件失败: {e}")

        print(f"视频生成完毕: {output_file_path}")
        return (output_file_path,)



NODE_CLASS_MAPPINGS = {
    "InfiniteTalkMultiImage": InfiniteTalkMultiImage,
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
    "MakeBatchFromIntList": "MakeBatchFromIntList",
    "GetIntByIndex": "GetIntByIndex",
    "GetFloatByIndex": "GetFloatByIndex",
    "MakeBatchFromFloatList": "MakeBatchFromFloatList",
    "InfiniteTalkEmbedsSlice": "InfiniteTalkEmbedsSlice",
    "AudioSmartSlice": "AudioSmartSlice",
    
    "VideoFromPathsAndAudio": "Video Synth (Path List + Audio)"

}