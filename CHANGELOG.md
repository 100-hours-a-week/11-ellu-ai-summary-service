# [1.6.0](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/compare/v1.5.1...v1.6.0) (2025-07-21)


### Features

* subtask 만들어진 답변 로깅 추가 ([abf1026](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/abf10263035f7d12cd7e4b9140cb16d638986150))


### Performance Improvements

* 오디오 API 디버깅 로그 추가 ([23b2896](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/23b28965c89bb7a3b2e7549697331acd42e49993))

## [1.5.1](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/compare/v1.5.0...v1.5.1) (2025-07-21)


### Bug Fixes

* 파싱 오류 수정 ([#227](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/227)) ([1cb3489](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/1cb348940064d94a61fe6077ecc60844b0d1877d))

# [1.5.0](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/compare/v1.4.0...v1.5.0) (2025-07-21)


### Bug Fixes

* api 변경 ([c8cc313](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/c8cc313d7cdc3409da784b161ab4f81f4585f0eb))
* api 변경 ([#208](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/208)) ([c5516d7](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/c5516d7b12a02c28855227aa2ab744ffce672982))
* gemini api 연결 오류 수정 ([#209](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/209)) ([8e6cb89](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/8e6cb89f49e048de8a1ca4076b01144a4b4f563d))
* gemini api 연결 오류 수정 ([#209](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/209)) ([#213](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/213)) ([6a2fbc4](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/6a2fbc42ec588357f05f8b3435f5934af974be1d))
* indentation 오류 수정, module import 수정 ([59585d9](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/59585d9cc037f8eebe85c108c89379587ee3f4df))
* minor change in processing metting note ([752e46c](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/752e46c65079bcfa5bc747ef561d264628732c3f))
* openai fallback([#222](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/222)) ([7c49b21](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/7c49b21716ca2158f94910be7afa2fac8ef52498))
* 회의록 업로드 기능 api 수정 ([833c50d](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/833c50dbc981ee84750a65ce1123f46a0f7cdc1e))


### Features

* pinecone subtext 검색 구현([#215](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/215)) ([e16706a](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/e16706af3760c1d4a72459e1d19a3b7f65f11559))
* 디비적용 및 fallback 구현([#218](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/218)) ([57c50b3](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/57c50b37f1513cc605556398ead6b7aa97a4d2fd))
* 역할 검색([#183](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/183)) ([d95d83e](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/d95d83e16a070059242ed61d0fcd1166226ba058))

# [1.4.0](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/compare/v1.3.4...v1.4.0) (2025-07-16)


### Bug Fixes

* ChromaDB 원본 로직 복원 및 올바른 트레이싱 추가 ([538d6b2](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/538d6b263328a5633e40f3fa6316741c388776bf))
* ChromaDB 컬렉션 존재 확인 후 작업 수행 ([a4cbdaa](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/a4cbdaa34f28f212d22d740c4be45199e228bc2c))
* requirements.txt torch install 추가 ([c79ee71](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/c79ee71b20a189bffbb171ca6d548a9a9cb810a9))
* vLLM commit 수정 ([1b8d9e1](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/1b8d9e1143e6521f106fda7de96c6e48b7bbd899))
* vllm json parsing 단순화 ([498e58a](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/498e58a68ecfa555cf6a4330023b322da65ba8fa))


### Features

* ChromaDB 트레이싱 추가 및 CUDA에서 Python slim 이미지로 변경 ([7c835b8](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/7c835b8cfe6d42109d59ea663ae1585ab44b962e))

## [1.3.4](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/compare/v1.3.3...v1.3.4) (2025-07-16)


### Bug Fixes

* release v1.3.4 ([20c2bca](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/20c2bca124988dcf6785e7d35f39bdf98427d20c))

## [1.3.3](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/compare/v1.3.2...v1.3.3) (2025-07-14)


### Bug Fixes

* vllm json parsing 단순화 ([dbf2ae3](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/dbf2ae34da902aab43f9491cb852df11b201ddce))

## [1.3.2](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/compare/v1.3.1...v1.3.2) (2025-07-14)


### Bug Fixes

* requirements.txt torch install 추가 ([5a6def4](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/5a6def4c050e7e26e1fa44b9084f561addd353e7))

## [1.3.1](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/compare/v1.3.0...v1.3.1) (2025-07-14)


### Bug Fixes

* vLLM commit 수정 ([5e98ce2](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/5e98ce268eaa98dad9fdd832cf447eeee17b30bf))

# [1.3.0](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/compare/v1.2.0...v1.3.0) (2025-07-08)


### Bug Fixes

* k8s deployment에서 모델 warmup endpoint 호출하기위해 curl 추가 ([7c8e652](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/7c8e65274fe74b984114e5e47633b8c275ea7bb4))
* k8s 환경을 위한 chromadb lazy loading 적용 ([d7d3cfd](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/d7d3cfd4fdc10df7450d91f8c96eab80e8df58af))
* main.py NoneType 수정 ([04cb923](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/04cb9232609f8df97ad5a3e695de7bfb61b86329))
* model lazy-loading 적용후  warmup 엔드포인트 추가 ([4708843](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/47088432e677e90674d7b1dcc5cfcca63f604d15))
* subtask오류 프롬프트 수정([#168](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/168)) ([db76880](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/db7688098919122eedbf02b06b858d83fe4e8f3d))
* 오류 부분 외 수정([#166](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/166)) ([97bdfe5](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/97bdfe56b6919c9e51d6a24fb65dc7db5dae2448))


### Features

* release ci 파이프라인 추가 ([5d3234a](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/5d3234aa6984e5ecd6b2809452ae94a163433767))
* s3, db 데이터 삭제기능([#144](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/144)) ([0a8ed51](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/0a8ed51dbc6a779af25f3414dfe291d14d2e71c6))
* wiki외 url크롤링([#100](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/100)) ([28282d5](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/28282d5aa7ff46fa652032a5ef53bd8b5d0c511b))
* 데이터 파이프라인 구축([#172](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/172)) ([2e67d89](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/2e67d89af498fbbdf9330b48f0eb6aceacfae435))
* 유저 데이터 저장 part 수정([#187](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/187)) ([93a57b9](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/93a57b994d102546d09b36a3c2f6548d70a73c76))

# [1.2.0](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/compare/v1.1.0...v1.2.0) (2025-06-20)


### Bug Fixes

* callback 함수 수정([#114](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/114)) ([094ab99](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/094ab99f73c7d1c37bbed61df482dfbd52fb3c38))
* container git error 해결 ([b4747d1](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/b4747d1b24d63fa0223ddb814b73ec7b760692d4))
* psycopg2 패키지 빌드 에러 수정 ([1449b86](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/1449b862f04dfe2dc34eb27fe29d07c244a19b13))
* revert stage.yml ([de781d3](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/de781d36649c15d101cf3573fe31661710b74960))
* stage.yml ([#147](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/147)) ([2263395](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/22633955654b07dd5c200a1cd321b27c288ebd02))
* url수신 오류 해결([#108](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/108)) ([db88f9b](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/db88f9b627c046f8ea0c459dd02d5035a32e4644))
* 빠져있던 prometheus 라이브러리 import문 추가 ([0e528a2](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/0e528a243f04a66b3509e15ff229b3c352b87675))
* 위키 가져오는 부분 비활성화 및 return 양식 수정([#110](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/110)) ([ba92266](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/ba92266f8e3682a4f84612c9f133657cec77db49))


### Features

* callback 함수 추가([#112](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/112)) ([59010f3](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/59010f36cd17db225e66d3e38584addcf2165c49))
* chroma 버전 관리 ([#104](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/104)) ([a0cec2e](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/a0cec2ecee5bdadf729948cef4d2f70b59b8d9fd))
* git 저장소 변경([#99](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/99)) ([b677e44](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/b677e4418a5324a9ee91545fe02e46480bb22552))
* JSON parsing 노드 및 검증 기능 개선 ([9bb9e61](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/9bb9e61ba392f2071a7ffdfa95745ccefc569e61)), closes [#127](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/127)
* JSON parsing 노드 및 검증 기능 개선 ([cb3475a](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/cb3475a785334e674a287243c51e6bbaa75006b4)), closes [#127](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/127)
* OpenTelemetry 트레이싱을 위한 클라이언트 및 SDK 추가 ([dff3f93](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/dff3f939d8da82127685129c59e9535a0fe72586))
* url기반 wiki 주소([#97](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/97)) ([0137bad](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/0137bada491ca622f1deb1ff62dbf3150ff944a5))
* wikifetcher클래스구현([#88](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/88)) ([65ad81a](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/65ad81a08bd3eb125002f9a4b8bf7bc9d56617bc))
* WikiRetreiver클래스 구현([#90](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/90)) ([d65af0e](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/d65af0ed1a3a04f391e4a0c4a24d86b997314d8e))
* 검증 노드 프롬프트 개선([#86](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/86)) ([fcffe37](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/fcffe37774efbba16e8173c69c3fde570b3f2471))
* 프롬프팅 미세 조정 및 오류 수정([#116](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/116)) ([2934c0a](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/2934c0a05b7a5d013b5fb4082230ec863531e18c))

# [1.1.0](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/compare/v1.0.0...v1.1.0) (2025-06-05)


### Bug Fixes

* 대소문자 버그 수정 ([#64](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/64)) ([da3f8fd](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/da3f8fd2f8b955ee46e2e74f3f87f35ed5d76a09))
* 모델 temperature strict positive float 버그 수정 ([e7a8b77](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/e7a8b77773d4aa7304f176188ccaf7dcb7b15667))


### Features

* gitignore 추가 및 meeting_chain 프롬프팅 수정([#62](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/62)) ([#65](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/65)) ([7e47e02](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/7e47e02375527d4ac214db317937983624b88b42))

# 1.0.0 (2025-05-11)


### Bug Fixes

* import embed_model 파일 경로 수정([#34](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/34)) ([b1ca814](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/b1ca8148acf540896309792f469c1f471acb5b01))
* import embed_model 파일 경로 수정([#34](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/34)) ([#35](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/35)) ([46ea74a](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/46ea74a8f4f0177b9095d90b43a4c657cfd975c6))
* Json parsing 수정 ([#53](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/53)) ([074a3a8](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/074a3a809805c555bd195ae6c096c8bb14e47371))
* Json parsing 수정 ([#53](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/53)) ([#54](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/54)) ([0162994](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/01629946d63c79a80849eb6dc9b23e4814af823f))
* wiki retriever 함수 수정([#41](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/41)) ([b61ca3a](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/b61ca3a1819ea3ecfa661a3b5d68f7604db63763))
* wiki retriever 함수 수정([#41](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/41)) ([#42](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/42)) ([b79d1b5](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/b79d1b5376d9407ecf6c75c23bf5c9d5bc74bf19))
* WikiSummarizer gpu 가속 버그 수정 ([344ebe9](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/344ebe9805666eba7ddc307ae7dcbf9fb6d3f5ef))
* WikiSummarizer self warm up 제거 ([21f1dc1](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/21f1dc13743847d80c4e15b36a34f48187be20f0))
* 벡터 디비 중복 호출 버그 수정 ([75b417c](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/75b417c87ea216da540d1ec23ac0e91543be87ec))
* 위키 체인 버그 수정 ([c136fb7](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/c136fb7ee33de6f6f09e8958c2c854682e584222))
* 회의록 전달 후 task 생성 api 버그 수정 ([#50](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/50)) ([180f3d6](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/180f3d66f67c9d4061399bba494941ee175bde15))


### Features

* fastapi 스켈레톤 생성 ([a03cf05](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/a03cf05b2311a9ea51a74e8b5cf74edf99460498))
* fastapi 스켈레톤 생성 ([#4](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/4)) ([2f95c35](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/2f95c35ec2eb733b7510d8e31aa0e774ee8bfd17))
* install 할 것 추가([#51](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/51)) ([3f42141](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/3f4214161a1ab79d82a2d05b8e581828936b9ce0))
* install 할 것 추가([#51](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/51)) ([#52](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/52)) ([4298903](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/4298903a6e24b622444e409ffa38b8daccd8ca6f))
* KoSimCSE 임베딩 모델 구현([#8](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/8)) ([d69b24d](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/d69b24d37e9f9248e8d06d6496ebdca3f6c08ab4))
* KoSimCSE 임베딩 모델 구현([#8](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/8)) ([#11](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/11)) ([9e354bc](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/9e354bc8b0d4e6a494341a7651186cdffeb7f712))
* semantic-release 적용 ([#56](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/56)) ([4f8cd40](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/4f8cd40843d363e6f29d768426f200abcb484feb))
* webhook API 구현([#46](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/46)) ([bb311cc](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/bb311ccff0f8db830f920b01ee92731de5464b14))
* webhook API 구현([#46](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/46)) ([#47](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/47)) ([1e6e7a1](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/1e6e7a12c9c09a824afd9dd72249ad1a4d03d915))
* wiki 요약정보 임베딩 후 metadata와 함께 ChromaDB에 저장하는 기능 구현([#23](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/23)) ([00ef70a](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/00ef70a23c42ed27f863e89b30a745523dd3afd1))
* wiki 요약정보 임베딩 후 metadata와 함께 ChromaDB에 저장하는 기능 구현([#23](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/23)) ([#26](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/26)) ([d4aac16](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/d4aac162c1fd531c8328816d4abf8f5203485eca))
* wiki정보 chromaDB 저장 기능 구현([#7](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/7)) ([6fad8d9](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/6fad8d9bafaf54df5627fd86dd82800bf90d9d1b))
* wiki정보 chromaDB 저장 기능 구현([#7](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/7)) ([#13](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/issues/13)) ([86216a7](https://github.com/100-hours-a-week/11-ellu-ai-summary-service/commit/86216a7ec0eb7b1465e726d0cc6edd2075e2da55))
