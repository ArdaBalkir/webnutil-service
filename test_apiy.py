import httpx

def test_amiup():
    response = httpx.get("http://localhost:8000/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_get_atlases():
    response = httpx.get("http://localhost:8000/atlases")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    assert "ABA_Mouse_CCFv3_2017_25um" in response.json()

def test_schedule_task():
    request_data = {
        "segmentation_path": "deepzoom-testing/webnutil_testdir/segmentations/",
        "alignment_json_path": "deepzoom-testing/webnutil_testdir/aba_mouse_ccfv3_2017_25um_2025-04-11_07-50-43.json",
        "colour": "[0, 0, 255]",
        "atlas_name": "ABA_Mouse_CCFv3_2017_25um",
        "output_path": "deepzoom-testing/webnutil_testdir/outputs",
        "token": "use_token_from_ebrains"
    }
    
    response = httpx.post("http://localhost:8000/schedule-task", json=request_data)
    assert response.status_code == 200
    assert "task_id" in response.json()
