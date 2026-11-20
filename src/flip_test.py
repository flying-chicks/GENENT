import numpy as np
import matplotlib.pyplot as plt

def filp_test(control_group, treatment_group, 
              n_permutations=10000, 
              alternative='two-sided',
              confidence_level=0.95):
    """
    执行置换检验
    
    参数:
    ----------
    control_group : array-like
        对照组数据
    treatment_group : array-like
        实验组数据
    n_permutations : int
        置换次数，默认10000
    alternative : str
        检验方向：'two-sided'（双尾）, 'greater'（单尾，实验组>对照组）, 
        'less'（单尾，实验组<对照组）
    confidence_level : float
        置信区间水平，默认0.95
    
    返回:
    -------
    dict : 包含统计量、p值、置信区间等结果
    """
    
    # 转换为numpy数组
    control = np.array(control_group)
    treatment = np.array(treatment_group)
    
    n_control = len(control)
    n_treatment = len(treatment)
    n_total = n_control + n_treatment
    
    # 观察到的统计量（均值差）
    observed_diff = np.mean(treatment) - np.mean(control)
    
    # 合并所有数据
    combined = np.concatenate([control, treatment])
    
    # 存储置换后的统计量
    perm_diffs = np.zeros(n_permutations)
    
    # 执行置换检验
    for i in range(n_permutations):
        # 随机打乱数据
        permuted = np.random.permutation(combined)
        
        # 分割为新的对照组和实验组
        perm_control = permuted[:n_control]
        perm_treatment = permuted[n_control:]
        
        # 计算置换后的均值差
        perm_diffs[i] = np.mean(perm_treatment) - np.mean(perm_control)
    
    # 计算p值
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    elif alternative == 'greater':
        p_value = np.mean(perm_diffs >= observed_diff)
    elif alternative == 'less':
        p_value = np.mean(perm_diffs <= observed_diff)
    else:
        raise ValueError("alternative参数必须是 'two-sided', 'greater', 或 'less'")
    
    # 计算置信区间（使用百分位数法）
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 - (1 - confidence_level) / 2) * 100
    
    # Bootstrap置信区间
    n_bootstrap = 5000
    bootstrap_diffs = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Bootstrap抽样
        boot_control = np.random.choice(control, size=n_control, replace=True)
        boot_treatment = np.random.choice(treatment, size=n_treatment, replace=True)
        bootstrap_diffs[i] = np.mean(boot_treatment) - np.mean(boot_control)
    
    ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
    ci_upper = np.percentile(bootstrap_diffs, upper_percentile)
    
    # 计算效应量（Cohen's d）
    pooled_std = np.sqrt(((n_control - 1) * np.var(control, ddof=1) + 
                          (n_treatment - 1) * np.var(treatment, ddof=1)) / 
                         (n_control + n_treatment - 2))
    cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0
    
    # 返回结果
    results = {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper),
        'cohens_d': cohens_d,
        'permutation_distribution': perm_diffs,
        'n_permutations': n_permutations,
        'alternative': alternative
    }
    
    return results


def plot_permutation_results(results, control, treatment, save_path):
    """
    绘制置换检验结果的可视化图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：置换分布和观察值
    axes[0].hist(results['permutation_distribution'], bins=50, 
                 density=True, alpha=0.7, color='skyblue', 
                 edgecolor='black', label='Permutation Distribution')
    axes[0].axvline(results['observed_diff'], color='red', 
                    linestyle='--', linewidth=2, 
                    label=f'Observed diff = {results["observed_diff"]:.3f}')
    axes[0].axvline(results['confidence_interval'][0], color='green', 
                    linestyle=':', linewidth=1.5, 
                    label=f'{int((1-results["p_value"])*100)}% CI')
    axes[0].axvline(results['confidence_interval'][1], color='green', 
                    linestyle=':', linewidth=1.5)
    axes[0].set_xlabel('Mean Difference')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Permutation Test Results ($p$-value = {results["p_value"]:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 右图：箱线图对比
    data_to_plot = [control, treatment]
    bp = axes[1].boxplot(data_to_plot, tick_labels=['Control', 'Treatment'], 
                         patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1].set_ylabel('Values')
    axes[1].set_title(f'Boxplot Comparison (Cohen\'s d = {results["cohens_d"]:.3f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


def print_results(results):
    """
    打印检验结果的详细报告
    """
    print("="*50)
    print("置换检验结果")
    print("="*50)
    print(f"观察到的均值差: {results['observed_diff']:.4f}")
    print(f"p值: {results['p_value']:.4f}")
    print(f"检验方向: {results['alternative']}")
    print(f"效应量 (Cohen's d): {results['cohens_d']:.4f}")
    print(f"置信区间 ({1-results['p_value']:.1%}): "
          f"[{results['confidence_interval'][0]:.4f}, "
          f"{results['confidence_interval'][1]:.4f}]")
    print(f"置换次数: {results['n_permutations']}")
    
    # 显著性判断
    alpha = 0.05
    if results['p_value'] < alpha:
        print(f"\n结论: 在显著性水平 α={alpha} 下，拒绝零假设。")
        print("两组之间存在显著差异。")
    else:
        print(f"\n结论: 在显著性水平 α={alpha} 下，不拒绝零假设。")
        print("没有足够的证据表明两组之间存在显著差异。")
    print("="*50)


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    
    # 对照组：均值50，标准差10
    control = np.random.normal(50, 10, 30)
    
    # 实验组：均值55，标准差10（有5单位的处理效应）
    treatment = np.random.normal(55, 10, 30)
    
    # 执行置换检验
    results = filp_test(control, treatment, 
                        n_permutations=10000,
                        alternative='two-sided')
    
    # 打印结果
    print_results(results)
    
    # 绘制可视化结果
    plot_permutation_results(results, control, treatment)
    
    # 也可以进行单尾检验
    print("\n单尾检验 (实验组 > 对照组):")
    results_one_tailed = filp_test(control, treatment, 
                                   n_permutations=10000,
                                   alternative='greater')
    print(f"p值: {results_one_tailed['p_value']:.4f}")