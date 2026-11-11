import numpy as np
import pandas as pd
import pytest
import ov_radiomics as rad


@pytest.fixture
def images(tmp_path):
    sitk = pytest.importorskip('SimpleITK')
    pytest.importorskip('radiomics')
    (tmp_path/'images').mkdir()
    (tmp_path/'masks').mkdir()
    for i in range(3):
        rng = np.random.RandomState(i)
        image = sitk.GetImageFromArray(rng.normal(30+i,5,(12,12,12)).astype(np.float32))
        voxels = np.zeros((12,12,12),dtype=np.uint8)
        voxels[3:9,3:9,3:9] = 1
        mask = sitk.GetImageFromArray(voxels)
        for value,folder in [(image,'images'),(mask,'masks')]:
            value.SetSpacing((1.2,1.2,2.0))
            sitk.WriteImage(value,str(tmp_path/folder/f'{i}.nii.gz'))
    return tmp_path


def test_real_extraction_and_geometry(images,tmp_path):
    import SimpleITK as sitk
    samples = rad.match_images(images)
    assert rad.check_images(samples).ok.all()
    params = {'imageType':{'Original':{}},'featureClass':{'firstorder':['Mean'],'shape':['VoxelVolume']},'setting':{'binWidth':5}}
    result = rad.extract(samples,params=params)
    assert len(result.features) == 3 and result.diagnostics.ok.all()
    assert result.features.original_shape_VoxelVolume.iloc[0] == pytest.approx(216*1.2*1.2*2)
    first = sitk.GetArrayFromImage(sitk.ReadImage(samples.image.iloc[0]))[3:9,3:9,3:9].mean()
    assert result.features.original_firstorder_Mean.iloc[0] == pytest.approx(first)
    result.save(tmp_path/'export')
    assert (tmp_path/'export/config.json').exists()
    mask = sitk.ReadImage(samples['mask'].iloc[0])
    mask.SetOrigin((100,0,0))
    sitk.WriteImage(mask,samples['mask'].iloc[0])
    assert not rad.check_images(samples).ok.iloc[0]
    with pytest.raises(RuntimeError,match='Geometry mismatch'):
        rad.extract(samples,params=params)
    recorded = rad.extract(samples,params=params,on_error='record')
    assert len(recorded.features) == 3
    assert not recorded.diagnostics.ok.iloc[0]
    assert pd.isna(recorded.features.original_firstorder_Mean.iloc[0])


def test_empty_roi_and_pairing(images):
    assert not rad.check_images(rad.match_images(images),roi_label=2).ok.any()
    (images/'masks/0.nii.gz').unlink()
    with pytest.raises(ValueError,match='Missing masks'):
        rad.match_images(images)
