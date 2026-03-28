param(
  [int]$CvId = 1,
  [string]$Question = "Goi y cong viec phu hop voi CV",
  [string]$BaseUrl = "http://127.0.0.1:8000"
)

$body = @{ cv_id = $CvId; question = $Question } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/v1/recommend/hybrid" -ContentType "application/json" -Body $body
