    PROGRAM MAIN
    
    USE ISO_C_BINDING
    USE modML

    IMPLICIT NONE
    
    CHARACTER*255 :: PATH
    INTEGER(C_INT) :: ID_1, ID_2, num_shape_dims, ierr
    
    REAL(C_FLOAT), dimension(15) :: input_1_tensor
    REAL(C_FLOAT), dimension(12) :: input_2_tensor
    REAL(C_FLOAT), dimension(3) :: output_1_tensor
    REAL(C_FLOAT), dimension(6) :: output_2_tensor

    INTEGER(C_INT64_T), dimension(2) :: input_1_tensor_shape, input_2_tensor_shape, output_1_tensor_shape, output_2_tensor_shape
    
    PATH = 'C:\Users\ihowell\Documents\nasa\tfexec\examples\fortran\model'
    ID_1=-1
    ID_2=-1
    ierr = ML_load_model(PATH, ID_1)
    WRITE(*,*) 'Model 1 Loaded: ', ID_1, ierr
    ierr = ML_load_model(PATH, ID_2)
    WRITE(*,*) 'Model 2 Loaded: ', ID_2, ierr

    ierr = ML_delete_model(ID_2)
    WRITE(*,*) 'Successfully deleted model'

    ierr = ML_register_output(ID_1, C_CHAR_"StatefulPartitionedCall:0" // C_NULL_CHAR)
    ierr = ML_register_output(ID_1, C_CHAR_"StatefulPartitionedCall:1" // C_NULL_CHAR)

    input_1_tensor = (/ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 /)
    input_2_tensor = (/ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 /)
    output_1_tensor = (/ -1.0, -1.0, -1.0 /)
    output_2_tensor = (/ -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 /)

    input_1_tensor_shape = (/ 3, 5 /)
    input_2_tensor_shape = (/ 3, 4 /)
    output_1_tensor_shape = (/ 3, 1 /)
    output_2_tensor_shape = (/ 3, 2 /)

    num_shape_dims = 2

    ierr = ML_provide_input(ID_1, C_CHAR_"serving_default_args_0:0" // C_NULL_CHAR, input_1_tensor_shape, input_1_tensor)
    ! ierr = ML_provide_input(ID_1, C_CHAR_"serving_default_args_1:0" // C_NULL_CHAR, input_2_tensor_shape, input_2_tensor)

    ierr = ML_predict(ID_1)
    if (ierr .ne. 0) then
        WRITE(*,*) 'Predict error:', ierr
        pause
        call exit(1)
    end if
        

    ierr = ML_retrieve_output(ID_1, "StatefulPartitionedCall:0", output_1_tensor_shape, output_1_tensor)
    ierr = ML_retrieve_output(ID_1, "StatefulPartitionedCall:1", output_2_tensor_shape, output_2_tensor)
    
    WRITE(*,*) 'Output 1', output_1_tensor
    WRITE(*,*) 'Output 2', output_2_tensor
    
    pause

    END PROGRAM MAIN