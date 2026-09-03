FROM radixark/vllm-glm53-flash:sm121-v8
ARG VLLM=/usr/local/lib/python3.12/dist-packages/vllm
COPY qwen3_dflash2.py $VLLM/model_executor/models/qwen3_dflash2.py
COPY dflash2/ $VLLM/v1/worker/gpu/spec_decode/dflash2/
COPY patch_registry_and_select.py /tmp/patch_registry_and_select.py
COPY patch_glm_aux_capture.py /tmp/patch_glm_aux_capture.py
COPY patch_kv_page_lcm2.py /tmp/patch_kv_page_lcm2.py
COPY patch_glm5_drafter_group.py /tmp/patch_glm5_drafter_group.py
COPY patch_prefix_cache_draft_group.py /tmp/patch_prefix_cache_draft_group.py
RUN python3 /tmp/patch_registry_and_select.py && python3 /tmp/patch_glm_aux_capture.py && python3 /tmp/patch_kv_page_lcm2.py && python3 /tmp/patch_glm5_drafter_group.py && python3 /tmp/patch_prefix_cache_draft_group.py &&     python3 -c "from vllm.model_executor.models.registry import ModelRegistry; assert \"DFlash2DraftModel\" in ModelRegistry.get_supported_archs(); print(\"v12 drafter-group build check OK\")"
