Load .so need to run some Creator's functions, as follow:
```
UpsamplePluginCreator::UpsamplePluginCreator
UpsamplePluginCreator::getPluginName
UpsamplePluginCreator::getPluginName
UpsamplePluginCreator::getPluginNamespace
UpsamplePluginCreator::getPluginVersion
```

when you build a engine, tensorrt will call those function:

```
UpsamplePluginCreator::getPluginName
UpsamplePluginCreator::createPlugin
UpsamplePlugin::UpsamplePlugin1
UpsamplePlugin::getNbOutputs
UpsamplePlugin::UpsamplePlugin1
UpsamplePlugin::getNbOutputs
UpsamplePlugin::getOutputDimensions
UpsamplePlugin::UpsamplePlugin1
UpsamplePlugin::supportsFormat
UpsamplePlugin::supportsFormat
UpsamplePlugin::supportsFormat
UpsamplePlugin::supportsFormat
UpsamplePlugin::supportsFormat
UpsamplePlugin::UpsamplePlugin1
UpsamplePlugin::supportsFormat
UpsamplePlugin::configureWithFormat
UpsamplePlugin::initialize
UpsamplePlugin::enqueue
```

when you save engine to a file, tensorrt will call those functions:

```
UpsamplePluginCreator::getPluginName
UpsamplePluginCreator::createPlugin
UpsamplePlugin::UpsamplePlugin1
UpsamplePlugin::getNbOutputs
UpsamplePlugin::UpsamplePlugin1
UpsamplePlugin::getNbOutputs
UpsamplePlugin::getOutputDimensions
UpsamplePlugin::UpsamplePlugin1
UpsamplePlugin::supportsFormat
UpsamplePlugin::supportsFormat
UpsamplePlugin::supportsFormat
UpsamplePlugin::supportsFormat
UpsamplePlugin::supportsFormat
UpsamplePlugin::UpsamplePlugin1
UpsamplePlugin::supportsFormat
UpsamplePlugin::configureWithFormat
UpsamplePlugin::initialize
UpsamplePlugin::getPluginType
UpsamplePlugin::getPluginVersion
UpsamplePlugin::getSerializationSize
UpsamplePlugin::serialize
UpsamplePlugin::getSerializationSize
UpsamplePlugin::getSerializationSize
```

when you load a engine, tensorrt will load those functions:
```
UpsamplePluginCreator::getPluginVersion
UpsamplePluginCreator::getPluginNamespace
UpsamplePluginCreator::deserializePlugin
UpsamplePlugin::UpsamplePlugin2
UpsamplePlugin::initialize
UpsamplePlugin::enqueue
```