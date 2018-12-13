UpsamplingPlugin

# How?

## 如何创建一个Layer Plugin?

一个Layer Plugin的实现必须包含两个class。一个是Plugin的实体，一个是Plugin的Creator。

具体要怎么实现，下面具体叙述。

+ Creator类，继承自IPluginCreator。
    + 构建函数：一般是初始化两个参数，mFC和mPluginAttributes。
    + getPluginName和getPluginVersion函数： 注册Creator所对应的Plugin的名称和版本。
    + getFieldNames函数：放回mFC变量。
    + creatorPlugin函数： 解析PluginFieldCollection，获取Plugin需要的参数。构建Plugin对象。
    + deserializePlugin函数： 从模型文件中载入engine的时候，构建Plugin所需要调用的函数。这个函数实现直接新建Plugin对象就行。
    + setPluginNamespace和getPluginNamespace，没有发现在什么阶段有调用这两个函数。
+ Plugin类，继承自IPluginV2。
    + 两个构建函数，一个用于在构建网络的时候，构建Plugin；另外一个是用于在反序列化的时候构建Plugin。（注意的是：如果这个层不支持没有参数输入的话，那么可以执行`UpsamplePlugin() = delete`删除默认构建函数）
    + getNbOutputs函数： 通常一个层是单输入单输出，所以这个函数直接`return 1`就行。
    + getOutputDimensions函数： 放回一个nvinfer1::Dims对象（或者它的子类），具体的值根据实际设置。
    + initialize，暂时没有用到。
    + teminate，暂时没有用到。
    + getWorkspaceSize，暂时没有用到,不清楚。
    + enqueue函数：用于正向传播，通常在这个函数调用cuda kernel。正常执行函数return 0。
    + getSerializationSize函数：当模型serialize的时候，需要保存到文件参数所占的具体空间。
    + serialize函数： 将Plugin参数保存至文件。
    + destroy函数： 删除当前类的this指针



## 如何让Tensorrt感知新的Layer Plugin？

在Plugin实现文件中调用`REGISTER_TENSORRT_PLUGIN`这个宏，用于注册一个Plugin Creator。例如：
```
REGISTER_TENSORRT_PLUGIN(UpsamplePluginCreator);
```

有了这个宏，当在.py文件中调用xxplugin.so的时候就会自动执行这个语句，然后就会在tensorrt中注册UpsamplePluginCreator的信息，可以用于创建新的Plugin,实际的效果就是在`trt.get_plugin_registry().plugin_creator_list`添加了一个`UpsamplePluginCreator`。

## Tensorrt如何调用一个Plugin？

### python的调用方式：

1. 获取tensorrt中的creator列表。代码如下：
    ```
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
    ```
2. 有了上面的列表，就可以根据名字匹配相应的 Plugin Creator，并且传入相应的参数，构建对应的plugin。代码如下：
    ```
    def get_upsample_plugin(plugin_name, sacle_factor=2, align_corners=False):
        plugin = None
        for plugin_creator in PLUGIN_CREATORS:
            if plugin_creator.name == plugin_name:
                scale_factor_field = trt.PluginField("scaleFactor", np.array([sacle_factor], dtype=np.int8), trt.PluginFieldType.INT8)
                align_corners_field = trt.PluginField("alignCorners", np.array([int(align_corners)], dtype=np.int8), trt.PluginFieldType.INT8)
                field_collection = trt.PluginFieldCollection([align_corners_field, scale_factor_field])
                plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
        return plugin
    ```
    + Note: 参数的载入tensorrt使用的是trt.PluginField，第一个参数是名字，第二个是参数的内存地址（buffer类型， 一般用numpy来实现），第三个是类型。名字和类型必须跟你在Creator中使用的一样，不然报错。
3. 创建好了Plugin，就可以用`network.add_plugin_v2`调用了。代码如下：
    ```
    upsample_layer = network.add_plugin_v2(inputs=[inputs], plugin=get_upsample_plugin("UpsamplePlugin", sacle_factor, align_corners))
    ```


### C++的调用方式：

TODO



# 附录

+ tensorrt在构建初次build engine以及engine serialize和deserialize的时候，调用的那些plugin的参数，可以参考`FunctionLoadLog.md`