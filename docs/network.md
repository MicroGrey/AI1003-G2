```mermaid
graph LR
    subgraph Input_Stage [输入阶段]
        Img_In[Input Image<br/>(3 Channels)] --> Replicate[Replicate x4<br/>(12 Channels)]
        Replicate --> FeaConv[Shallow Feature Ext<br/>BSConvU/S]
    end

    subgraph Deep_Feature_Extraction [深度特征提取 (Body)]
        FeaConv --> ESDB_1[ESDB Block 1]
        ESDB_1 --> ESDB_2[ESDB Block 2]
        ESDB_2 -.-> ESDB_N[... ESDB Block N]
        
        %% 特征聚合 (Feature Aggregation)
        ESDB_1 -- append --> Concat_List
        ESDB_2 -- append --> Concat_List
        ESDB_N -- append --> Concat_List
    end

    subgraph Reconstruction_Stage [重建阶段]
        Concat_List[Concat All Block Outputs] --> Fusion_1x1[Conv 1x1]
        Fusion_1x1 --> Act_SiLU[Activation: SiLU]
        Act_SiLU --> Fusion_3x3[BSConv 3x3]
        
        %% 全局残差连接
        FeaConv -- Global Residual --> Add_Global((+))
        Fusion_3x3 --> Add_Global
        
        Add_Global --> Upsampler[Upsampler<br/>Conv 3x3 + PixelShuffle]
        Upsampler --> Img_Out[Output Image]
    end

    style Input_Stage fill:#f9f,stroke:#333,stroke-width:2px
    style Deep_Feature_Extraction fill:#e1f5fe,stroke:#333,stroke-width:2px
    style Reconstruction_Stage fill:#e8f5e9,stroke:#333,stroke-width:2px
```