import os
import io
import re
import time
import base64
import requests
import shutil
import time

import numpy
import PIL

import folder_paths

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
        import torch

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
        import torch

        # 只有一个或没有 → 直接返回
        if len(int_list) <= 1:
            return (int_list,)

        # 转为 tensor batch
        # int_list 是 Python int 列表，需要转成 tensor
        int_tensor = torch.tensor(int_list, dtype=torch.int32)

        return (int_tensor,)

class InfiniteTalkMultiImage():
    """
    ComfyUI 节点：
    - 输入：audio_duration, fps, 最多20张图片及每张图片的 start_time
    - 输出：images（按 start_time 排序），bigLoopFrames 列表，real_start_times（基于 bigLoopFrames 的实际开始时间，单位秒）
    - 新增：text_list_input 文本框，按行切分并返回 text_list
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "audio_duration": ("FLOAT", {"default": 8.0, "min": 0.01, "step": 0.001}),
                "prompt_list_input": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {}
        }

        # 添加最多 20 张图片与 start_time
        for i in range(1, 21):
            inputs["optional"][f"image{i}"] = ("IMAGE",)
            inputs["required"][f"start_time{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.001})

        return inputs

    OUTPUT_IS_LIST = (True, True, True, True)
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "STRING",)  
    RETURN_NAMES = ("image_list", "frame_count_list", "real_start_time_list", "prompt_list")
    FUNCTION = "calculate_big_loops"
    CATEGORY = "InfiniteTalk"

    @staticmethod
    def calculate_big_loops(**kwargs):
        fps = 25
        audio_duration = float(kwargs.get("audio_duration", 0.0))
        total_frames = int(round(audio_duration * fps))

        raw_text = kwargs.get("prompt_list_input", "")
        # 按行 split，去掉空行
        text_list = [line.strip() for line in raw_text.split("\n") if line.strip()]

        # ================= 收集图片与 start time =================
        pairs = []
        for i in range(1, 21):
            img = kwargs.get(f"image{i}", None)
            st = kwargs.get(f"start_time{i}", None)
            if img is not None:
                st_val = float(st) if st is not None else 0.0
                pairs.append((st_val, img))

        if not pairs:
            return [], [], [], text_list

        # 按 start_time 排序
        pairs.sort(key=lambda x: x[0])
        input_start_times = [p[0] for p in pairs]
        images = [p[1] for p in pairs]

        # 计算每段 framesNeeded
        frames_needed = []
        for idx, st in enumerate(input_start_times):
            end_t = input_start_times[idx + 1] if idx < len(input_start_times) - 1 else audio_duration
            dur = max(0.0, end_t - st)
            fn = int(round(dur * fps))
            frames_needed.append(fn)

        # bigLoopFrames 规则：81 + 72*n >= frames_needed
        big_loop_frames = []
        for fn in frames_needed:
            if fn <= 81:
                cap = 81
            else:
                n = -(-(fn - 81) // 72)
                cap = 81 + 72 * n
            big_loop_frames.append(cap)

        # 若总帧数超过音频长度，压缩最后一个
        s = sum(big_loop_frames)
        if s > total_frames:
            allowed_last = total_frames - sum(big_loop_frames[:-1])
            big_loop_frames[-1] = max(1, allowed_last)

        # 计算真实开始帧
        real_start_frames = []
        cur = 0
        for cap in big_loop_frames:
            real_start_frames.append(cur)
            cur += cap

        # 转为秒
        real_start_times = [f / fps for f in real_start_frames]
        real_start_times.append(audio_duration)

        # ================= 对齐 text_list =================
        n_images = len(images)
        n_texts = len(text_list)
        if n_texts < n_images:
            # 不够用最后一行补齐
            if text_list:
                text_list.extend([text_list[-1]] * (n_images - n_texts))
            else:
                text_list = [""] * n_images
        elif n_texts > n_images:
            # 多余的截断
            text_list = text_list[:n_images]

        return images, big_loop_frames, real_start_times, text_list



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



NODE_CLASS_MAPPINGS = {
    "InfiniteTalkMultiImage": InfiniteTalkMultiImage,
    "MakeBatchFromIntList": MakeBatchFromIntList,
    "GetIntByIndex": GetIntByIndex,
    "GetFloatByIndex": GetFloatByIndex,
    "MakeBatchFromFloatList": MakeBatchFromFloatList,
    "InfiniteTalkEmbedsSlice": InfiniteTalkEmbedsSlice,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InfiniteTalkMultiImage": "InfiniteTalkMultiImage",
    "MakeBatchFromIntList": "MakeBatchFromIntList",
    "GetIntByIndex": "GetIntByIndex",
    "GetFloatByIndex": "GetFloatByIndex",
    "MakeBatchFromFloatList": "MakeBatchFromFloatList",
    "InfiniteTalkEmbedsSlice": "InfiniteTalkEmbedsSlice",

}