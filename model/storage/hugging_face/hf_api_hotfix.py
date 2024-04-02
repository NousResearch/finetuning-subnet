# huggingface_hub contains a bug where SafeTensorsInfo is instantiated with
# kwargs containing an unexpected key 'sharded', leading to an exception.
#
# By replacing the SafeTensorsInfo class with a subclassed version, the
# 'sharded' key can be removed.

from huggingface_hub import hf_api

class SafeTensorsInfo_fixed(hf_api.SafeTensorsInfo):
    orig = hf_api.SafeTensorsInfo
    def __init__(self,**kwargs):
        kwargs.pop('sharded',None)
        SafeTensorsInfo_fixed.orig.__init__(self,**kwargs)

hf_api.SafeTensorsInfo = SafeTensorsInfo_fixed
