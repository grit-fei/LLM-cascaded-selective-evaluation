import jsonlines
import itertools

from cascaded_evaluation.cascaded_classifier import CascadedClassifier
from model.task_spec import BINARY_TASK

if __name__ == "__main__":
    model_names = ["Qwen3.5-27B","Qwen3.5-35B-A3B","Qwen3.6-27B","Qwen3.6-35B-A3B"]
    calibration_filenames = [
        "./result_605/Qwen3.5-27B-Q4_K_M.gguf.calibration.jsonl"
        "./result_605/Qwen3.5-35B-A3B-Q4_K_M.gguf.calibration.jsonl"
        "./result_605/Qwen3.6-27B-Q4_K_M.gguf.calibration.jsonl"
        "./result_605/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf.calibration.jsonl"
    ]
    test_filenames = [
        "./result_605/Qwen3.5-27B-Q4_K_M.gguf.test.jsonl"
        "./result_605/Qwen3.5-35B-A3B-Q4_K_M.gguf.test.jsonl"
        "./result_605/Qwen3.6-27B-Q4_K_M.gguf.test.jsonl"
        "./result_605/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf.test.jsonl"
    ]
    
    model_to_calibration = dict(zip(model_names,calibration_filenames))
    model_to_test = dict(zip(model_names,test_filenames))
    
    calibration_samples = {}
    for model_name, input_filename in zip(model_names,test_filenames):
        with jsonlines.open(input_filename) as f:
            test_samples[model_name] = list(f)
    
    test_samples = {}
    for model_name, input_filename in zip(model_names, test_filenames):
        with jsonlines.open(input_filename) as f:
            test_samples[model_name] = list(f)
    
    all_permutations = list(itertools.permutations(model_names))
    print(f"共{len(all_permutations)}种排列组合\n")
    print("="*60)
    
    results = []
    
    for perm in all_permutations:
        perm = list(perm)
        print(f"\n排列顺序:{perm}")
        print("-"*60)
        
        cascaded_classifier = CascadedClassifier(
            perm,
            calibration_samples=calibration_samples,
            alpha=0.1
            delta=0.1
            task_type=BINARY_TASK
        )
        
        evaluator_compostion, selective_acc, coverage, evaluators = cascaded_classifier.apply_decision_rule(
            test_samples,
            perm
        )
        
        print("lambda_hats", cascaded_classifier.lambda_hats)
        print(f"Evaluator Composition: {evaluator_compostion}")
        print(f"Acc: {selective_acc}")
        print(f"Coverage: {coverage}")
        print(f"Evaluated samples: {int((evaluators >= 0).sum())}")
        
        results.append({
            "permutation":perm,
            "lambda_hats": cascaded_classifier.lambda_hats,
            "evaluator_composition": evaluator_compostion,
            "selective_acc": selective_acc,
            "coverage": coverage,
            "evaluated_samples": int((evaluators >= 0).sum()),
        })
        
    from cascaded_evalution.util import merge_data
    merged_cal = merge_data(calibration_samples, model_names, task_type=BINARY_TASK)
    merged_test = merge_data(test_samples,model_names,task_type=BINARY_TASK)
    print("\n"+"="*60)
    print("所有排列结果汇总：")
    print("="*60)
    for r in results:
        print(f"排列":{r['permutation']})
        print(f" Acc:{r['selective_acc']:.4f}| Coverage:{r['coverage']:.4f}| Evaluated:{r['evaluated_samples']}")
        print(f" Evaluator Composition:{r['evaluator_composition']}")
        print()