Search.setIndex({"docnames": ["api_ref_dtypes", "api_ref_intro", "api_ref_kernel", "api_ref_quantization", "api_ref_sparsity", "dtypes", "generated/torchao.dtypes.AffineQuantizedTensor", "generated/torchao.dtypes.to_affine_quantized", "generated/torchao.dtypes.to_nf4", "generated/torchao.quantization.Int4WeightOnlyGPTQQuantizer", "generated/torchao.quantization.Int4WeightOnlyQuantizer", "generated/torchao.quantization.SmoothFakeDynQuantMixin", "generated/torchao.quantization.SmoothFakeDynamicallyQuantizedLinear", "generated/torchao.quantization.int4_weight_only", "generated/torchao.quantization.int8_dynamic_activation_int4_weight", "generated/torchao.quantization.int8_dynamic_activation_int8_weight", "generated/torchao.quantization.int8_weight_only", "generated/torchao.quantization.quantize_", "generated/torchao.quantization.smooth_fq_linear_to_inference", "generated/torchao.quantization.swap_linear_with_smooth_fq_linear", "generated/torchao.sparsity.PerChannelNormObserver", "generated/torchao.sparsity.WandaSparsifier", "generated/torchao.sparsity.apply_fake_sparsity", "getting-started", "index", "overview", "performant_kernels", "quantization", "serialization", "sg_execution_times", "sparsity", "tutorials/index", "tutorials/sg_execution_times", "tutorials/template_tutorial"], "filenames": ["api_ref_dtypes.rst", "api_ref_intro.rst", "api_ref_kernel.rst", "api_ref_quantization.rst", "api_ref_sparsity.rst", "dtypes.rst", "generated/torchao.dtypes.AffineQuantizedTensor.rst", "generated/torchao.dtypes.to_affine_quantized.rst", "generated/torchao.dtypes.to_nf4.rst", "generated/torchao.quantization.Int4WeightOnlyGPTQQuantizer.rst", "generated/torchao.quantization.Int4WeightOnlyQuantizer.rst", "generated/torchao.quantization.SmoothFakeDynQuantMixin.rst", "generated/torchao.quantization.SmoothFakeDynamicallyQuantizedLinear.rst", "generated/torchao.quantization.int4_weight_only.rst", "generated/torchao.quantization.int8_dynamic_activation_int4_weight.rst", "generated/torchao.quantization.int8_dynamic_activation_int8_weight.rst", "generated/torchao.quantization.int8_weight_only.rst", "generated/torchao.quantization.quantize_.rst", "generated/torchao.quantization.smooth_fq_linear_to_inference.rst", "generated/torchao.quantization.swap_linear_with_smooth_fq_linear.rst", "generated/torchao.sparsity.PerChannelNormObserver.rst", "generated/torchao.sparsity.WandaSparsifier.rst", "generated/torchao.sparsity.apply_fake_sparsity.rst", "getting-started.rst", "index.rst", "overview.rst", "performant_kernels.rst", "quantization.rst", "serialization.rst", "sg_execution_times.rst", "sparsity.rst", "tutorials/index.rst", "tutorials/sg_execution_times.rst", "tutorials/template_tutorial.rst"], "titles": ["torchao.dtypes", "<code class=\"docutils literal notranslate\"><span class=\"pre\">torchao</span></code> API Reference", "torchao.kernel", "torchao.quantization", "torchao.sparsity", "Dtypes", "AffineQuantizedTensor", "to_affine_quantized", "to_nf4", "Int4WeightOnlyGPTQQuantizer", "Int4WeightOnlyQuantizer", "SmoothFakeDynQuantMixin", "SmoothFakeDynamicallyQuantizedLinear", "int4_weight_only", "int8_dynamic_activation_int4_weight", "int8_dynamic_activation_int8_weight", "int8_weight_only", "quantize", "smooth_fq_linear_to_inference", "swap_linear_with_smooth_fq_linear", "PerChannelNormObserver", "WandaSparsifier", "apply_fake_sparsity", "Getting Started", "Welcome to the torchao Documentation", "Overview", "Performant Kernels", "Quantization", "Serialization", "Computation times", "Sparsity", "&lt;no title&gt;", "Computation times", "Template Tutorial"], "terms": {"thi": [1, 6, 12, 13, 14, 20, 21, 22, 28, 33], "section": 1, "introduc": 1, "dive": 1, "detail": 1, "how": [1, 6, 13, 28], "integr": [1, 28], "pytorch": [1, 24, 33], "optim": [1, 17], "your": [1, 17, 24], "machin": 1, "learn": [1, 13, 33], "model": [1, 14, 17, 18, 19, 21, 22, 24], "sparsiti": [1, 20, 21, 22, 24, 28], "quantiz": [1, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 24, 28], "dtype": [1, 6, 7, 8, 10, 17, 24, 28], "kernel": [1, 6, 13, 17], "tba": [2, 5, 23, 25, 26, 27, 30], "class": [6, 9, 10, 11, 12, 20, 21, 28], "torchao": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 28], "layout_tensor": 6, "aqtlayout": 6, "block_siz": [6, 7, 8], "tupl": [6, 7, 21], "int": [6, 7, 8, 10, 21], "shape": 6, "size": [6, 13, 14, 28], "quant_min": [6, 7], "option": [6, 7, 10, 17, 18, 19, 21], "none": [6, 7, 17, 18, 19, 21], "quant_max": [6, 7], "zero_point_domain": [6, 7, 13, 17], "zeropointdomain": [6, 7, 13], "stride": 6, "sourc": [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 31, 33], "affin": [6, 17], "tensor": [6, 7, 8, 13, 17, 21, 28, 33], "subclass": [6, 12, 17, 20, 28], "mean": 6, "we": [6, 17, 28], "float": [6, 7, 13, 17, 19, 21, 28], "point": [6, 13, 28], "an": [6, 21, 24], "transform": 6, "quantized_tensor": 6, "float_tensor": 6, "scale": [6, 11, 12, 18, 19], "zero_point": [6, 13], "The": [6, 18, 19, 21, 28], "repres": [6, 21, 28], "look": 6, "extern": 6, "regardless": 6, "intern": 6, "represent": [6, 13], "s": 6, "type": [6, 9, 10, 13, 28], "orient": 6, "field": 6, "serv": 6, "gener": [6, 31, 33], "layout": [6, 13], "storag": 6, "data": [6, 28], "e": [6, 17, 28], "g": [6, 17, 28], "store": [6, 20], "plain": 6, "int_data": 6, "pack": 6, "format": 6, "depend": [6, 28], "devic": [6, 9, 10, 28], "oper": 6, "granular": [6, 13, 14], "element": 6, "share": 6, "same": 6, "qparam": 6, "when": 6, "input": [6, 17, 21], "dimens": 6, "ar": [6, 13, 17, 21, 28], "us": [6, 13, 14, 17, 21, 22, 24, 28], "per": [6, 12, 13, 14, 15, 16, 21], "torch": [6, 10, 12, 13, 17, 18, 19, 22, 28, 33], "minimum": 6, "valu": [6, 11, 12, 18, 21], "specifi": [6, 21], "deriv": 6, "from": [6, 14, 17, 28, 29, 32, 33], "maximum": [6, 18], "domain": [6, 13], "should": [6, 12, 20, 21], "eitehr": 6, "integ": [6, 13], "zero": [6, 13, 21], "ad": [6, 21], "dure": [6, 19], "subtract": 6, "unquant": 6, "default": [6, 17, 18, 19], "input_quant_func": 6, "callabl": [6, 17], "function": [6, 12, 17, 20, 21, 22, 24, 28], "object": 6, "take": [6, 12, 17, 20], "output": [6, 33], "float32": [6, 28], "dequant": [6, 13], "given": 6, "return": [6, 17, 18, 19, 28], "classmethod": [6, 12], "implement": [6, 12, 28], "aten_ops_or_torch_fn": 6, "decor": 6, "aten": [6, 13], "op": [6, 13, 17], "__torch_dispatch__": 6, "user": [6, 33], "pass": [6, 12, 20], "list": [6, 19, 21], "__torch_function__": 6, "singl": 6, "mytensor": 6, "_implement": 6, "nn": [6, 12, 17, 18, 19, 28], "linear": [6, 12, 13, 14, 15, 16, 17, 19, 22, 28], "def": [6, 17, 28], "_": 6, "func": 6, "arg": [6, 11, 12, 21], "kwarg": [6, 11, 12, 20, 21, 22], "perform": [6, 11, 12, 18, 20], "convers": [6, 17], "A": [6, 20], "infer": [6, 12, 18, 28], "argument": [6, 17], "self": [6, 11, 12, 28], "If": [6, 18, 21], "alreadi": 6, "ha": 6, "correct": 6, "otherwis": 6, "copi": [6, 21, 28], "desir": 6, "here": [6, 28], "wai": 6, "call": [6, 12, 17, 20, 28], "non_block": 6, "fals": [6, 13, 17, 18, 21, 28], "memory_format": 6, "preserve_format": 6, "memori": 6, "tri": 6, "convert": [6, 12, 17], "asynchron": 6, "respect": 6, "host": 6, "possibl": 6, "cpu": [6, 28], "pin": 6, "cuda": [6, 9, 10, 28], "set": [6, 11, 12, 17, 18, 21], "new": [6, 17], "creat": 6, "even": 6, "match": 6, "other": [6, 21, 28, 33], "exampl": [6, 17, 21, 28, 29, 31, 32, 33], "randn": [6, 28], "2": [6, 13, 17, 22, 33], "initi": [6, 28], "float64": 6, "0": [6, 9, 11, 12, 17, 19, 21, 28, 29, 32, 33], "5044": 6, "0005": 6, "3310": 6, "0584": 6, "cuda0": 6, "true": [6, 7, 9, 10, 17, 18, 28], "input_float": 7, "mapping_typ": 7, "mappingtyp": 7, "target_dtyp": 7, "ep": 7, "scale_dtyp": 7, "zero_point_dtyp": [7, 17], "preserve_zero": [7, 13, 17], "bool": [7, 10, 17, 18], "layout_typ": [7, 15], "layouttyp": 7, "plainlayouttyp": [7, 15], "64": [8, 9, 13, 28], "scaler_block_s": 8, "256": [8, 10, 13], "blocksiz": 9, "128": [9, 13], "percdamp": 9, "01": 9, "groupsiz": [9, 10, 17], "inner_k_til": [9, 10, 13], "8": [9, 10, 13], "padding_allow": [9, 10], "precis": 10, "bfloat16": [10, 17, 28], "set_debug_x_absmax": [11, 12], "x_running_abs_max": [11, 12], "which": [11, 12, 28], "lead": [11, 12], "smooth": [11, 12], "all": [11, 12, 20, 21, 22, 28, 29, 31], "ones": [11, 12, 21], "alpha": [11, 12, 19], "5": [11, 12, 19, 21, 33], "enabl": [11, 12], "benchmark": [11, 12, 18], "without": [11, 12], "calibr": [11, 12], "replac": [12, 19], "dynam": [12, 14, 15], "token": [12, 14, 15], "activ": [12, 14, 15, 18, 21], "channel": [12, 15, 16, 20], "weight": [12, 13, 14, 15, 16, 17, 21, 28], "base": [12, 21], "smoothquant": [12, 18, 19], "forward": [12, 20, 28], "x": [12, 17, 28, 33], "defin": [12, 20, 21], "comput": [12, 20, 21], "everi": [12, 20], "overridden": [12, 20], "although": [12, 20], "recip": [12, 20], "need": [12, 20, 21, 28], "within": [12, 20], "one": [12, 20], "modul": [12, 17, 18, 19, 20, 21, 28], "instanc": [12, 17, 20, 28], "afterward": [12, 20], "instead": [12, 13, 20], "sinc": [12, 20, 28], "former": [12, 20], "care": [12, 20, 28], "run": [12, 17, 18, 20, 33], "regist": [12, 20], "hook": [12, 20], "while": [12, 20, 21], "latter": [12, 20], "silent": [12, 20], "ignor": [12, 20], "them": [12, 20], "from_float": 12, "mod": 12, "fake": 12, "version": 12, "note": [12, 21], "requir": 12, "to_infer": 12, "calcul": [12, 18], "prepar": [12, 18, 21], "group_siz": [13, 14, 17], "appli": [13, 14, 15, 16, 17], "uint4": [13, 17], "onli": [13, 16, 17, 28], "asymmetr": [13, 14, 17], "group": [13, 14], "layer": [13, 15, 16, 18, 19, 21, 22], "tensor_core_til": 13, "speedup": 13, "tinygemm": [13, 17], "target": [13, 21], "int4mm": 13, "_weight_int4pack_mm": 13, "main": 13, "differ": [13, 28], "algorithm": 13, "compar": [13, 21], "more": [13, 14, 24], "tradit": 13, "follow": 13, "1": [13, 17, 21, 28, 29, 32, 33], "doe": 13, "have": [13, 21], "exactli": 13, "choose_qparams_affin": 13, "pleas": 13, "relev": [13, 33], "code": [13, 31, 33], "quantize_affin": 13, "dequantize_affin": 13, "about": [13, 28], "paramet": [13, 14, 17, 18, 19, 21, 28], "chosen": 13, "control": [13, 14, 21], "smaller": [13, 14, 28], "fine": [13, 14], "grain": [13, 14], "choic": 13, "32": [13, 14, 17, 28], "int4": [13, 14, 17, 28], "mm": [13, 17], "4": [13, 22, 28], "int8": [14, 15, 16, 17], "symmetr": [14, 15, 16], "produc": 14, "executorch": [14, 17], "backend": 14, "current": [14, 17, 19, 21], "did": 14, "support": [14, 28], "lower": 14, "flow": [14, 22], "yet": 14, "quantize_": [17, 28], "apply_tensor_subclass": 17, "filter_fn": 17, "str": [17, 19, 21], "set_inductor_config": 17, "modifi": [17, 21], "inplac": [17, 21], "fulli": [17, 19], "qualifi": [17, 19], "name": [17, 19, 21], "want": [17, 28], "whether": 17, "automat": [17, 33], "recommend": 17, "inductor": 17, "config": [17, 21], "import": [17, 28, 33], "some": [17, 21], "predefin": 17, "method": [17, 21], "correspond": [17, 28], "execut": [17, 29, 32], "path": 17, "also": [17, 28], "customiz": 17, "int8_dynamic_activation_int4_weight": 17, "int8_dynamic_activation_int8_weight": 17, "compil": 17, "int4_weight_onli": [17, 28], "int8_weight_onli": 17, "quant_api": [17, 28], "m": [17, 28], "sequenti": 17, "1024": [17, 28], "write": 17, "own": 17, "you": [17, 21, 28, 33], "can": [17, 28], "add": [17, 33], "manual": 17, "constructor": 17, "to_affine_quant": 17, "groupwis": 17, "apply_weight_qu": 17, "lambda": 17, "int32": 17, "15": 17, "1e": 17, "6": 17, "apply_weight_quant_to_linear": 17, "requires_grad": 17, "under": [17, 24], "block0": 17, "submodul": 17, "fqn": [17, 21], "isinst": 17, "debug_skip_calibr": 18, "each": [18, 20], "smoothfakedynamicallyquantizedlinear": [18, 19], "contain": [18, 19], "debug": 18, "skip_fqn_list": 19, "cur_fqn": 19, "equival": 19, "skip": [19, 21], "being": 19, "process": [19, 33], "factor": 19, "custom": 20, "observ": 20, "l2": 20, "norm": [20, 21], "buffer": 20, "x_orig": 20, "sparsity_level": 21, "semi_structured_block_s": 21, "wanda": 21, "sparsifi": [21, 28], "prune": [21, 22, 24], "propos": 21, "http": 21, "arxiv": 21, "org": 21, "ab": 21, "2306": 21, "11695": 21, "awar": 21, "remov": 21, "product": 21, "magnitud": 21, "three": 21, "variabl": 21, "number": 21, "spars": 21, "block": 21, "out": 21, "level": 21, "dict": 21, "parametr": 21, "preserv": 21, "origin": [21, 28], "deepcopi": 21, "squash_mask": 21, "params_to_keep": 21, "params_to_keep_per_lay": 21, "squash": 21, "mask": 21, "appropri": 21, "either": 21, "sparse_param": 21, "attach": 21, "kei": [21, 33], "save": [21, 28], "param": 21, "specif": [21, 28], "string": 21, "xdoctest": 21, "local": 21, "undefin": 21, "don": 21, "t": 21, "ani": 21, "hasattr": 21, "submodule1": 21, "keep": 21, "linear1": [21, 28], "foo": 21, "bar": 21, "submodule2": 21, "linear42": 21, "baz": 21, "print": [21, 28, 33], "42": 21, "24": 21, "update_mask": 21, "tensor_nam": 21, "statist": 21, "retriev": 21, "first": 21, "act_per_input": 21, "Then": 21, "metric": 21, "matrix": 21, "across": 21, "whole": 21, "simul": 22, "It": 22, "ao": 22, "open": 24, "librari": [24, 28], "provid": 24, "nativ": 24, "our": 24, "develop": 24, "content": 24, "come": 24, "soon": 24, "question": 28, "peopl": 28, "especi": 28, "describ": [28, 33], "work": 28, "tempfil": 28, "toylinearmodel": 28, "__init__": 28, "n": 28, "k": 28, "super": 28, "bia": 28, "linear2": 28, "example_input": 28, "batch_siz": 28, "in_featur": 28, "eval": 28, "f": 28, "get_model_size_in_byt": 28, "mb": [28, 29, 32], "ref": 28, "namedtemporaryfil": 28, "state_dict": 28, "seek": 28, "load": 28, "meta": 28, "m_load": 28, "so": 28, "check": 28, "befor": 28, "load_state_dict": 28, "assign": 28, "after": 28, "re": 28, "assert": 28, "equal": 28, "To": 28, "just": 28, "becaus": 28, "techniqu": 28, "like": 28, "thing": 28, "chang": 28, "structur": 28, "For": 28, "float_weight1": 28, "float_weight2": 28, "quantized_weight1": 28, "quantized_weight2": 28, "typic": 28, "go": [28, 33], "techinqu": 28, "util": 28, "abov": 28, "see": 28, "reduct": 28, "around": 28, "4x": 28, "0625": 28, "reason": 28, "avoid": 28, "mai": 28, "fit": 28, "updat": 28, "affinequantizedtensor": 28, "No": 28, "verifi": 28, "properli": 28, "affine_quantized_tensor": 28, "00": [29, 32], "004": [29, 32, 33], "total": [29, 32, 33], "file": [29, 32], "galleri": [29, 31, 33], "mem": [29, 32], "templat": [29, 31, 32], "tutori": [29, 31, 32], "tutorials_sourc": 29, "template_tutori": [29, 32, 33], "py": [29, 32, 33], "download": [31, 33], "python": [31, 33], "tutorials_python": 31, "zip": [31, 33], "jupyt": [31, 33], "notebook": [31, 33], "tutorials_jupyt": 31, "sphinx": [31, 33], "end": 33, "full": 33, "author": 33, "firstnam": 33, "lastnam": 33, "what": 33, "item": 33, "3": 33, "prerequisit": 33, "v2": 33, "gpu": 33, "why": 33, "topic": 33, "link": 33, "research": 33, "paper": 33, "walk": 33, "through": 33, "below": 33, "rand": 33, "7998": 33, "1532": 33, "4973": 33, "1979": 33, "0402": 33, "1262": 33, "6189": 33, "7451": 33, "3599": 33, "8280": 33, "0365": 33, "0570": 33, "7993": 33, "5081": 33, "7737": 33, "practic": 33, "test": 33, "knowledg": 33, "nlp": 33, "scratch": 33, "summar": 33, "concept": 33, "cover": 33, "highlight": 33, "takeawai": 33, "link1": 33, "link2": 33, "time": 33, "script": 33, "minut": 33, "second": 33, "ipynb": 33}, "objects": {"torchao.dtypes": [[6, 0, 1, "", "AffineQuantizedTensor"], [7, 2, 1, "", "to_affine_quantized"], [8, 2, 1, "", "to_nf4"]], "torchao.dtypes.AffineQuantizedTensor": [[6, 1, 1, "", "dequantize"], [6, 1, 1, "", "implements"], [6, 1, 1, "", "to"]], "torchao.quantization": [[9, 0, 1, "", "Int4WeightOnlyGPTQQuantizer"], [10, 0, 1, "", "Int4WeightOnlyQuantizer"], [11, 0, 1, "", "SmoothFakeDynQuantMixin"], [12, 0, 1, "", "SmoothFakeDynamicallyQuantizedLinear"], [13, 2, 1, "", "int4_weight_only"], [14, 2, 1, "", "int8_dynamic_activation_int4_weight"], [15, 2, 1, "", "int8_dynamic_activation_int8_weight"], [16, 2, 1, "", "int8_weight_only"], [17, 2, 1, "", "quantize_"], [18, 2, 1, "", "smooth_fq_linear_to_inference"], [19, 2, 1, "", "swap_linear_with_smooth_fq_linear"]], "torchao.quantization.SmoothFakeDynQuantMixin": [[11, 1, 1, "", "set_debug_x_absmax"]], "torchao.quantization.SmoothFakeDynamicallyQuantizedLinear": [[12, 1, 1, "", "forward"], [12, 1, 1, "", "from_float"], [12, 1, 1, "", "set_debug_x_absmax"], [12, 1, 1, "", "to_inference"]], "torchao": [[4, 3, 0, "-", "sparsity"]], "torchao.sparsity": [[20, 0, 1, "", "PerChannelNormObserver"], [21, 0, 1, "", "WandaSparsifier"], [22, 2, 1, "", "apply_fake_sparsity"]], "torchao.sparsity.PerChannelNormObserver": [[20, 1, 1, "", "forward"]], "torchao.sparsity.WandaSparsifier": [[21, 1, 1, "", "prepare"], [21, 1, 1, "", "squash_mask"], [21, 1, 1, "", "update_mask"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:function", "3": "py:module"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "function", "Python function"], "3": ["py", "module", "Python module"]}, "titleterms": {"torchao": [0, 1, 2, 3, 4, 24], "dtype": [0, 5], "api": [1, 24], "refer": [1, 24], "python": 1, "kernel": [2, 26], "quantiz": [3, 17, 27], "sparsiti": [4, 30], "affinequantizedtensor": 6, "to_affine_quant": 7, "to_nf4": 8, "int4weightonlygptqquant": 9, "int4weightonlyquant": 10, "smoothfakedynquantmixin": 11, "smoothfakedynamicallyquantizedlinear": 12, "int4_weight_onli": 13, "int8_dynamic_activation_int4_weight": 14, "int8_dynamic_activation_int8_weight": 15, "int8_weight_onli": 16, "smooth_fq_linear_to_infer": 18, "swap_linear_with_smooth_fq_linear": 19, "perchannelnormobserv": 20, "wandasparsifi": 21, "apply_fake_spars": 22, "get": 23, "start": 23, "welcom": 24, "document": 24, "overview": [25, 33], "perform": 26, "serial": 28, "deseri": 28, "flow": 28, "what": 28, "happen": 28, "when": 28, "an": 28, "optim": 28, "model": 28, "comput": [29, 32], "time": [29, 32], "templat": 33, "tutori": 33, "step": 33, "option": 33, "addit": 33, "exercis": 33, "conclus": 33, "further": 33, "read": 33}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})