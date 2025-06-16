! Resource on Fortran to C Strings: http://fortranwiki.org/fortran/show/Generating+C+Interfaces

!--------------------------ML Import Function Modules------------------- 
   MODULE modML
       USE ISO_C_BINDING, ONLY : C_INT64_T
      !EXTERNAL ML_load_model 
      !!DEC$ ATTRIBUTES DLLIMPORT :: ML_load_model
   
      implicit none
      
      interface
      integer function C_tfexec__load_model(model_path, model_id) bind(C, name="tfexec__load_model")
          USE ISO_C_BINDING, ONLY : C_INT, C_CHAR
          character(kind=C_CHAR), dimension(*), intent(in)  :: model_path
          integer(kind=C_INT), intent(out) :: model_id
      end function
      end interface
      
      interface
      integer function C_tfexec__register_output(model_id, op_name) bind(C, name="tfexec__register_output")
          USE ISO_C_BINDING, ONLY : C_INT, C_CHAR
          integer(kind=C_INT), intent(in) :: model_id
          character(kind=C_CHAR), dimension(*), intent(in) :: op_name
      end function
      end interface
      
      interface
      integer function C_tfexec__delete_model(model_id) bind(C, name="tfexec__delete_model")
          USE ISO_C_BINDING, ONLY : C_INT
          integer(C_INT), intent(in) :: model_id
      end function      
      end interface   

      interface
      integer function C_tfexec__provide_input(model_id, op_name, num_shape_dims, shape, data_in) bind(C, name="tfexec__provide_input")
          USE ISO_C_BINDING, ONLY : C_INT, C_CHAR, C_INT64_T, C_FLOAT
          integer(C_INT), intent(in) :: model_id
          character(C_CHAR), dimension(*), intent(in) :: op_name
          integer(C_INT), intent(in) :: num_shape_dims
          integer(C_INT64_t), dimension(*), intent(in) :: shape
          real(C_FLOAT), dimension(*), intent(in) :: data_in
      end function
      end interface

      interface
      integer function C_tfexec__predict(model_id) bind(C, name="tfexec__predict")
          USE ISO_C_BINDING, ONLY : C_INT
          integer(C_INT), intent(in) :: model_id
      end function
      end interface

      interface
      integer function C_tfexec__retrieve_output(model_id, op_name, num_shape_dims, shape, data_out) bind(C, name="tfexec__retrieve_output")
          USE ISO_C_BINDING, ONLY : C_CHAR, C_INT, C_INT64_T, C_FLOAT
          integer(C_INT), intent(in) :: model_id
          character(C_CHAR), dimension(*), intent(in)  :: op_name
          integer(C_INT), intent(in) :: num_shape_dims
          integer(C_INT64_T), dimension(*), intent(in) :: shape
          real(C_FLOAT), dimension(*), intent(inout) :: data_out
      end function
      end interface

      interface
      integer function C_tfexec__retrieve_output_no_check(model_id, op_name, data_out) bind(C, name="tfexec__retrieve_output_no_check")
          USE ISO_C_BINDING, ONLY : C_CHAR, C_INT, C_INT64_T, C_FLOAT
          integer(C_INT), intent(in) :: model_id
          character(C_CHAR), dimension(*), intent(in)  :: op_name
          real(C_FLOAT), dimension(*), intent(inout) :: data_out
      end function
      end interface

      interface ML_load_model
          module procedure tfexec__load_model
      end interface

      interface ML_register_output
          module procedure tfexec__register_output
      end interface

      interface ML_delete_model
          module procedure tfexec__delete_model
      end interface
      
      interface ML_provide_input
          module procedure tfexec__provide_input
      end interface

      interface ML_predict
          module procedure tfexec__predict
      end interface

      interface ML_retrieve_output
          module procedure tfexec__retrieve_output
      end interface
      
      interface ML_retrieve_output_no_check
          module procedure tfexec__retrieve_output_no_check
      end interface
      
      public :: ML_load_model
      public :: ML_register_output
      public :: ML_delete_model
      public :: ML_predict
      public :: ML_retrieve_output
      public :: ML_retrieve_output_no_check
      
      CONTAINS
    
          integer function tfexec__load_model(model_path, model_id)
            !DEC$ ATTRIBUTES DLLEXPORT :: tfexec__load_model
            USE ISO_C_BINDING, ONLY : C_INT, C_CHAR, C_NULL_CHAR
            character*(*), intent(in) :: model_path
            integer(C_INT), intent(inout) :: model_id
            tfexec__load_model=C_tfexec__load_model(TRIM(model_path) // C_NULL_CHAR, model_id)
          end function

          integer function tfexec__register_output(model_id, op_name)
            !DEC$ ATTRIBUTES DLLEXPORT :: tfexec__register_output
            USE ISO_C_BINDING, ONLY : C_INT, C_CHAR, C_NULL_CHAR
            integer(C_INT), intent(in) :: model_id
            character*(*), intent(in) :: op_name
            tfexec__register_output=C_tfexec__register_output(model_id, TRIM(op_name) // C_NULL_CHAR)
          end function
    
          integer function tfexec__delete_model(model_id)
            !DEC$ ATTRIBUTES DLLEXPORT :: tfexec__delete_model
            USE ISO_C_BINDING, ONLY : C_INT
            integer(C_INT), intent(in) :: model_id
            tfexec__delete_model=C_tfexec__delete_model(model_id)
          end function

          integer function tfexec__provide_input(model_id, op_name, shape, data_in)
            !DEC$ ATTRIBUTES DLLEXPORT :: tfexec__provide_input
            USE ISO_C_BINDING, ONLY : C_CHAR, C_INT, C_INT64_T, C_FLOAT, C_NULL_CHAR
            integer(C_INT), intent(in) :: model_id
            character*(*), intent(in) :: op_name
            integer(C_INT64_T), dimension(:), intent(in) :: shape
            real(C_FLOAT), dimension(*), intent(in) :: data_in
            tfexec__provide_input=C_tfexec__provide_input(model_id, TRIM(op_name) // C_NULL_CHAR, size(shape), shape, data_in)
          end function

          integer function tfexec__predict(model_id)
            !DEC$ ATTRIBUTES DLLEXPORT :: tfexec__predict
            USE ISO_C_BINDING, ONLY : C_INT
            integer(C_INT), intent(in) :: model_id
            tfexec__predict=C_tfexec__predict(model_id)
          end function

          integer function tfexec__retrieve_output(model_id, op_name, shape, data_out)
            !DEC$ ATTRIBUTES DLLEXPORT :: tfexec__retrieve_output
            USE ISO_C_BINDING, ONLY : C_CHAR, C_INT, C_INT64_T, C_FLOAT, C_NULL_CHAR
            integer(C_INT), intent(in) :: model_id
            character*(*), intent(in)  :: op_name
            integer(C_INT64_T), dimension(:), intent(in) :: shape
            real(C_FLOAT), dimension(*), intent(inout) :: data_out
            tfexec__retrieve_output=C_tfexec__retrieve_output(model_id, TRIM(op_name) // C_NULL_CHAR, size(shape), shape, data_out)
          end function

          integer function tfexec__retrieve_output_no_check(model_id, op_name, data_out)
            !DEC$ ATTRIBUTES DLLEXPORT :: tfexec__retrieve_output_no_check
            USE ISO_C_BINDING, ONLY : C_INT, C_FLOAT, C_NULL_CHAR
            integer(C_INT), intent(in) :: model_id
            character*(*), intent(in)  :: op_name
            real(C_FLOAT), dimension(*), intent(inout) :: data_out
            tfexec__retrieve_output_no_check=C_tfexec__retrieve_output_no_check(model_id, TRIM(op_name) // C_NULL_CHAR, data_out)
          end function

    
   END MODULE modML
!-------------------------------------------------------------------- 