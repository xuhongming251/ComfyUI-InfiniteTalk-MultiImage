

 
1. 支持InfiniteTalk 多图输入。
2. 支持每张图指定开始时间点。
3. 支持为为每张图指定提示词。
4. 支持真无限长视频生成，显存和内存不受视频时长而递增，每批次直接写磁盘，最后合并视频。
5. 完全兼容KJ节点，不入侵不修改kj节点。


>> 实现输入一个音频，提供多张图片，为每张图片指定开始时间，
并且为每张图片指定单独提示词控制。**
一次运行，视频直出。

---

1. Support multi-image input for InfiniteTalk.
2. Allow each image to have its own start time.
3. Allow each image to have its own prompt.
4. Support true infinite-length video generation — GPU/CPU memory should not increase with video duration. Each batch is written directly to disk, and the final video is merged at the end.
5. Fully compatible with all KJ nodes, no modifications or intrusive changes required.

>> The system takes an audio file, multiple images, and lets you set a start time and custom prompt for each image.
Run it once, and the full video comes out directly.

![](workflow.png)
