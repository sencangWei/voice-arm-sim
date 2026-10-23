"""Generate a labelled synthetic Mandarin test WAV using the system eSpeak NG.

This exercises the audio API; it does not substitute for a human microphone test.
API: https://github.com/espeak-ng/espeak-ng/blob/master/src/include/espeak-ng/speak_lib.h
"""
import argparse
import ctypes as c
import ctypes.util
from pathlib import Path
import wave


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="把苹果放进蓝色托盘")
    parser.add_argument("--output", default="runs/synthetic-command.wav")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    library = ctypes.util.find_library("espeak-ng")
    if not library:
        raise RuntimeError("需要系统 libespeak-ng1 和 espeak-ng-data，仅用于合成测试音频")
    lib = c.CDLL(library)
    lib.espeak_Initialize.argtypes = [c.c_int, c.c_int, c.c_char_p, c.c_int]
    sample_rate = lib.espeak_Initialize(2, 0, None, 0)
    if sample_rate <= 0:
        raise RuntimeError("eSpeak NG 初始化失败")
    chunks = []
    callback_type = c.CFUNCTYPE(c.c_int, c.POINTER(c.c_short), c.c_int, c.c_void_p)
    @callback_type
    def receive(samples, count, events):
        if samples and count:
            chunks.append(c.string_at(samples, count * 2))
        return 0
    lib.espeak_SetSynthCallback.argtypes = [callback_type]
    lib.espeak_SetSynthCallback(receive)
    lib.espeak_SetVoiceByName.argtypes = [c.c_char_p]
    if lib.espeak_SetVoiceByName(b"cmn") != 0:
        raise RuntimeError("缺少普通话 cmn 声音")
    lib.espeak_SetParameter(1, 145, 0)
    lib.espeak_Synth.argtypes = [c.c_char_p, c.c_size_t, c.c_uint, c.c_int,
                                 c.c_uint, c.c_uint, c.c_void_p, c.c_void_p]
    text = args.text.encode("utf-8")
    if lib.espeak_Synth(text, len(text) + 1, 0, 1, 0, 1, None, None) != 0:
        raise RuntimeError("语音合成失败")
    lib.espeak_Terminate()
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        wav.writeframes(b"".join(chunks))
    print(f"合成测试语音（非真人）: {output}; 指令: {args.text}")


if __name__ == "__main__":
    main()
