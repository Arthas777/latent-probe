"""
Run CoT-SFT inference on GSM8k-Aug test set (1319 problems).
Saves per-problem: answer, correctness, chain length (tokens generated).
Uses CoLaR's CoT-SFT checkpoint (same backbone, LoRA-based CoT training).

Usage: python run_cot_sft_inference.py [--device cuda:0]
"""
import os, sys, json, torch, argparse
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from src.utils.utils import get_position_ids_from_attention_mask


COT_CKPT = os.path.join(project_root, 'checkpoints/logs/cot/qsa-gsm/llama-1b-cot/checkpoints/epoch7__step12056__monitor0.560.ckpt')
LLM_PATH = os.path.join(project_root, 'models/llms/Llama-3.2-1B-Instruct')


def load_cot_model(device):
    ckpt = torch.load(COT_CKPT, map_location='cpu', weights_only=False)
    hparams = ckpt['hyper_parameters']
    model_kwargs = hparams['model_kwargs']

    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    llm = AutoModelForCausalLM.from_pretrained(LLM_PATH).to(device)
    llm.resize_token_embeddings(len(tokenizer))
    llm.generation_config.pad_token_id = tokenizer.pad_token_id
    llm.generation_config.eos_token_id = tokenizer.eos_token_id

    if model_kwargs.get('do_lora', False):
        lora_config = LoraConfig(**model_kwargs['lora_config'])
        llm = get_peft_model(llm, peft_config=lora_config)

    state_dict = ckpt['state_dict']
    llm_state = {k.replace('llm.', ''): v for k, v in state_dict.items() if k.startswith('llm.')}
    llm.load_state_dict(llm_state, strict=False)
    llm.eval()

    answer_gen_config = dict(model_kwargs['answer_generation_config'])
    return llm, tokenizer, answer_gen_config


def extract_answer_cot(text, tokenizer):
    """Extract answer and chain length from CoT-SFT output."""
    sep = "###"
    if sep in text:
        parts = text.split(sep)
        chain = parts[0] if len(parts) > 0 else ""
        answer_part = parts[-1] if len(parts) > 1 else ""
    else:
        chain = text
        answer_part = text

    if 'Answer:' in answer_part:
        answer_str = answer_part.split('Answer:')[-1].strip()
    else:
        answer_str = answer_part.strip()

    chain_tokens = len(tokenizer.encode(chain, add_special_tokens=False))
    return answer_str, chain_tokens


def clean_answer(s):
    import re
    s = re.sub(r'<\|[^|]*\|>', '', s)
    s = s.strip().rstrip('.').replace(',', '').lower()
    return s


def check_correct(pred_str, gt_str):
    pred = clean_answer(pred_str)
    gt = clean_answer(str(gt_str))
    try:
        return abs(float(pred) - float(gt)) < 1e-4
    except (ValueError, TypeError):
        return pred == gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading CoT-SFT model on {device}...")
    llm, tokenizer, gen_config = load_cot_model(device)

    thinking_separator = "###"
    question_template = "Question: {} Let's think step by step:"
    speed_template = "(Thinking speed: {})"

    test_path = os.path.join(project_root, 'datasets/text_reasoning/gsm/test.json')
    with open(test_path) as f:
        data = json.load(f)
    print(f"Running CoT-SFT inference on {len(data)} problems...")

    results = []
    for pi, example in enumerate(tqdm(data, desc="CoT-SFT inference")):
        question = example['question']
        gt = str(example['answer']).strip()

        text = question_template.format(question) + speed_template.format(1) + thinking_separator
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False, padding=False)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            pred_ids = llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
            )

        gen_ids = pred_ids[0, input_ids.shape[1]:]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=False)
        total_gen_tokens = len(gen_ids)

        answer_str, chain_tokens = extract_answer_cot(decoded, tokenizer)
        correct = check_correct(answer_str, gt)

        results.append({
            "idx": pi,
            "correct": bool(correct),
            "answer": answer_str,
            "total_tokens": total_gen_tokens,
            "chain_tokens": chain_tokens,
        })

        if (pi + 1) % 100 == 0:
            acc = sum(r['correct'] for r in results) / len(results)
            avg_tok = sum(r['total_tokens'] for r in results) / len(results)
            print(f"  [{pi+1}] acc={acc:.4f}, avg_tokens={avg_tok:.1f}")

    acc = sum(r['correct'] for r in results) / len(results)
    avg_tokens = sum(r['total_tokens'] for r in results) / len(results)
    print(f"\nFinal: acc={acc:.4f}, avg_tokens={avg_tokens:.1f}")

    os.makedirs('./routing_results', exist_ok=True)
    output = {
        "config": {"model": "CoT-SFT-1B", "n_problems": len(results)},
        "accuracy": float(acc),
        "avg_tokens": float(avg_tokens),
        "per_problem": results,
    }
    with open('./routing_results/cot_sft_inference.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to routing_results/cot_sft_inference.json")


if __name__ == "__main__":
    main()
