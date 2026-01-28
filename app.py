# -*- coding: utf-8 -*-
"""ResNet18.a1_in1k 图像分类 WebUI 演示界面（不加载真实模型权重）。"""
import gradio as gr


def run_classify(image, top_k_text):
    """模拟图像分类：仅展示界面与结果区域，不执行模型推理。"""
    if image is None:
        return "请上传一张图片。\n\n加载模型后，将在此显示 ImageNet 千类 Top-K 预测结果。"
    try:
        k = int(float(top_k_text or 5))
    except (ValueError, TypeError):
        k = 5
    k = max(1, min(10, k))
    lines = ["【演示模式】未加载模型，以下为示例输出格式：\n"]
    for i in range(k):
        lines.append(f"  Top-{i+1}: 黄金猎犬 (golden retriever) — 92.00%")
    return "\n".join(lines)


with gr.Blocks(title="ResNet18.a1_in1k 图像分类 WebUI") as demo:
    gr.Markdown("# ResNet18.a1_in1k WebUI\n\n图像分类可视化界面（支持上传图片与 Top-K 预测，演示模式不加载真实权重）")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传图像", type="pil")
            top_k = gr.Textbox(label="Top-K 数量", value="5", placeholder="1–10")
            run_btn = gr.Button("开始分类（演示）", variant="primary")
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="分类结果", lines=12)
    run_btn.click(fn=run_classify, inputs=[input_image, top_k], outputs=output_text)
    gr.Markdown("---\n**模型说明**：ResNet18.a1_in1k 为基于 ResNet Strikes Back A1 配方在 ImageNet-1k 上预训练的模型，本界面用于演示加载与图像分类流程，未实际下载权重。")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
