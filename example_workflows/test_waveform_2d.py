# -*- coding: utf-8 -*-
import torchaudio

# โหลดไฟล์เสียงที่ต้องการทดสอบ
waveform, sr = torchaudio.load("testTHmono.wav")

# แสดงรูปร่างของ waveform
print("✅ waveform shape:", waveform.shape)
